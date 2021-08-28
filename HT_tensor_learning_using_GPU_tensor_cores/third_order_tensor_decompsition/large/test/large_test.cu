#include<iostream>
#include<fstream>
#include <assert.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda_fp16.h>

using namespace std;

void printTensor(float *d_des,long m,long n,long l){
    float *des = new float[m*n*l]();
    cudaMemcpy(des,d_des,sizeof(float)*m*n*l,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int k = 0;k<l;k++){
        for(int i = 0;i<n;i++){
            for(int j = 0;j<m;j++){
                cout<<des[k*m*n+i*m+j]<<" ";
            }
            cout<<endl;
        }
        cout<<"~~~~~~~~~~~~~~~~"<<endl;
    }
    delete[] des;des=nullptr;

}
__global__ void sqrt_gpu(float *d_A,float *d_B,int m,int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    const int temp = blockDim.x*gridDim.x;

    while(i<n*m)
    {
        int row = i%m;
        int col = i/m;
        if(row == col)
        {
            d_B[i] = sqrt(d_A[row]);           
        }
        else
        {
            d_B[i]=0;
        }
        
        i+=temp;
    }
     __syncthreads();
}


int main()
{
	int a = 3;
    int b = 2;

    float *A = new float[a*b];
    for(unsigned i = 0; i < a*b; ++i) {
        /* code */
        A[i]=i+1;
    }
	float *d_A;
	cudaMalloc((void**)&d_A,sizeof(float)*a*b);
    cudaMemcpy(d_A,A,sizeof(float)*a*b,cudaMemcpyHostToDevice);


	//A初始化为正实数的矩阵  （能否保证为正定矩阵）————> 正定矩阵的特征值为正，可以开根号得到实数奇异值

	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
  	//cudaStream_t stream = NULL;
 	//syevjInfo_t syevj_params = NULL;

    //cudaStream_t stream2 = NULL; 

    //cudaStream_t stream = NULL;

	float alpha = 1.0;
	float beta = 0.0;

/*	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	curandGenerateUniform(gen,d_A,a*b);*/


	//需要 A = (n*n) * n 的U
	//1、 A.T * A 的特征向量为 V，特征值开根号为S

	float *d_ATA;
	cudaMalloc((void**)&d_ATA, sizeof(float)*b*b);


	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,b,a,&alpha,d_A,a,d_A,a,&beta,d_ATA,b);
   // printTensor(d_ATA,b,b,1);

	//d_ATA 特征分解


    float *d_W = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    int  lwork = 0;

    cudaMalloc ((void**)&d_W, sizeof(float) * b);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnSsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        b,
        d_ATA,
        b,
        d_W,
        &lwork);
     cudaMalloc((void**)&d_work, sizeof(float)*lwork);
     cusolverDnSsyevd(
        cusolverH,jobz,uplo,b,d_ATA,b,
        d_W,d_work,lwork,devInfo); 

     //这里求得的SVD 的V 与实际上SVD的V是不同的，后者自带转置，且降序；前者截断时取列再转置，且升序

     float *d_S,*d_ST,*d_AT;
     cudaMalloc((void**)&d_S,sizeof(float)*a*b);
     cudaMalloc((void**)&d_ST,sizeof(float)*b*a);
     
     cudaMalloc((void**)&d_AT,sizeof(float)*b*a);
    printTensor(d_ATA,b*b,1,1);
     sqrt_gpu<<<512,512>>>(d_W,d_S,a,b);
     cudaDeviceSynchronize();
     //S * V (暂时不知道是 V  还是  V.T) 
     // a*b  b*b

     //下一步需要对 d_ST转置，所以这里直接求转置后的 d_ST,即 V.T * (d_S).T = d_ST  b*b   *  b*a

     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,b,a,b,&alpha,d_ATA,b,d_S,a,&beta,d_ST,b);

     // A= U * d_ST ==> (d_ST).T  *  U.T  = A.T   (b*a) * (a*a) = (b*a)
     //此时的  d_ST已经是 (b*a), 直接求
     //首先 QR分解   (LU分解的方程求解需要 方阵)
     

     cublasSgeam( handle,
                  CUBLAS_OP_T, CUBLAS_OP_N,
                          b, a,
                          &alpha,
                          d_A, a,
                          &beta,
                          d_A, a,
                          d_AT,b);
     printTensor(d_ST,b,a,1);


     int *devInfo2 = NULL;
    float *d_work2 = NULL;
    int  lwork_geqrf = 0;
    int  lwork_ormqr = 0;
    int  lwork2 = 0;
     float *d_tau = NULL;
    cudaMalloc ((void**)&d_tau, sizeof(float) * a);

    cudaMalloc ((void**)&devInfo2, sizeof(int));
     cusolverDnSgeqrf_bufferSize(
        cusolverH,
        b,
        a,
        d_ST,
        b,
        &lwork_geqrf);
     cusolverDnSormqr_bufferSize(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,  //?
        b,
        a,
        b,
        d_ST,
        b,
        d_tau,
        d_AT,
        a,
        &lwork_ormqr);
     lwork2 = (lwork_geqrf > lwork_ormqr)? lwork_geqrf : lwork_ormqr;
     cudaMalloc((void**)&d_work2, sizeof(double)*lwork2);
     cusolverDnSgeqrf(cusolverH, 
        b, a, d_ST, b, 
        d_tau, d_work2, lwork2, devInfo2);

     cusolverDnSormqr(cusolverH,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,
        b,a,b,d_ST,b,
        d_tau,d_AT,b,
        d_work2,lwork2,devInfo2);

     cublasStrsm(
         handle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N, 
         CUBLAS_DIAG_NON_UNIT,
         b,
         a,
         &alpha,
         d_ST,
         b,
         d_AT,
         b);
     cudaDeviceSynchronize();
    // printTensor(d_AT,a*b,1,1);
     //此结果应该是 U.T, 升序
     float *d_temp,*d_tmp2;
     cudaMalloc((void**)&d_temp,sizeof(float)*a*b);
     cudaMalloc((void**)&d_tmp2,sizeof(float)*a*b);

     cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,a,b,&alpha,d_AT,b,&beta,d_temp,b,d_tmp2,a);
      //printTensor(d_tmp2,a*b,1,1);

//直接 SVD 分解 A，比较特征分解的结果 与 SVD的 V；  
// A = U S V.T   下面直接对A svd分解

     float *d_U = NULL;
     float *d_Ss = NULL;
    float *d_VT = NULL;
    int *devInfo3 = NULL;
    float *d_work3 = NULL;
    float *d_rwork = NULL;
    int lwork3 = 0;
    cudaMalloc ((void**)&d_Ss  , sizeof(float)*b);
    cudaMalloc((void**)&d_U,sizeof(float)*a*a);
    cudaMalloc ((void**)&d_VT , sizeof(float)*b*b);
     cudaMalloc ((void**)&devInfo, sizeof(int));

     cusolverDnSgesvd_bufferSize(
        cusolverH,
        a,
        b,
        &lwork3 );
     cudaMalloc((void**)&d_work3 , sizeof(double)*lwork3);
     signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolverDnSgesvd (
        cusolverH,
        jobu,
        jobvt,
        a,
        b,
        d_A,
        a,
        d_Ss,
        d_U,
        a,  // ldu
        d_VT,
        b, // ldvt,
        d_work3,
        lwork3,
        d_rwork,
        devInfo3);
    //printTensor(d_U,a*b,1,1);
    printTensor(d_VT,b*b,1,1);

}