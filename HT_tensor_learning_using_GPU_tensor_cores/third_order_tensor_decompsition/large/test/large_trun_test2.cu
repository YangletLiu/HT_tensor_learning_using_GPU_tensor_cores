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
typedef float dt;
using namespace std;

void  gentuTensor1(dt *X,long a,long b,long c,long r1,long r2,long r3)
{
    dt *A,*B,*C,*G;
    cudaHostAlloc((void**)&A,sizeof(dt)*a*r1,0);
    cudaHostAlloc((void**)&B, sizeof(dt)*b*r2,0);
    cudaHostAlloc((void**)&C, sizeof(dt)*c*r3,0);
    cudaHostAlloc((void**)&G, sizeof(dt)*r1*r2*r3,0);
    srand(123);

    for(long long i=0;i<a*r1;i++)
    {
        A[i] = rand()*0.1/(RAND_MAX*0.1);
    }
    for(long long i=0;i<b*r2;i++)
    {
        B[i] = rand()*0.1/(RAND_MAX*0.1);
    }
    for(long long i=0;i<c*r3;i++)
    {
        C[i] = rand()*0.1/(RAND_MAX*0.1);
    }
    for(long long i=0;i<r1*r2*r3;i++)
    {
        G[i] = rand()*0.1/(RAND_MAX*0.1);
    }

    dt * d_A,*d_B,*d_C,*d_X,*d_G;
    cudaMalloc((void**)&d_A,sizeof(dt)*a*r1);
    cudaMalloc((void**)&d_B,sizeof(dt)*b*r2);
    cudaMalloc((void**)&d_C,sizeof(dt)*c*r3);
    cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);
    cudaMalloc((void**)&d_G,sizeof(dt)*r2*r1*r3);

    cudaMemcpyAsync(d_A, A,sizeof(dt)*a*r1,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_B, B,sizeof(dt)*b*r2,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_C, C,sizeof(dt)*c*r3,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_G, G,sizeof(dt)*r1*r2*r3,cudaMemcpyHostToDevice,0);
    dt *d_AG,*d_AGB;
    cudaMalloc((void**)&d_AG,sizeof(dt)*a*r2*r3);
    cudaMalloc((void**)&d_AGB,sizeof(dt)*b*a*r3);
    dt alpha = 1.0;
    dt beta =0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3,r1,&alpha,d_A,a,d_G,r1,&beta,d_AG,a);
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,&alpha,d_AG,a,a*r2,d_B,b,0,&beta,d_AGB,a,a*b,r3);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,&alpha,d_AGB,a*b,d_C,c,&beta,d_X,a*b);
    cudaMemcpyAsync(X,d_X,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,  0);
    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_G);
    cudaFree(d_AGB);
    cudaFree(d_AG);
    cudaFree(d_X);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(G);
    cublasDestroy(handle);
}
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

__global__ void transmission(dt *d_A,dt *d_B,long a,long b)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;
    while(i<a*b)
    {
      long col=i/a+1;
      long row=i%a;
      d_B[a*(b-col)+row]=d_A[i];
      i+=temp;
    }
 __syncthreads();
}
__global__ void sqrt_gpu2(float *d_A,float *d_B,int b,int k)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    const int temp = blockDim.x*gridDim.x;

    while(i<k*k)
    {
        int row = i%k;
        int col = i/k;
        if(row == col)
        {
            d_B[i] = 1.0/(sqrt(d_A[b-k+row]));     
                
        }
        else
        {
            d_B[i]=0;
        }
        
        i+=temp;
    }
     __syncthreads();
}
__global__ void fuHao(float*d_A,float *d_B,int m,int n)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
    const int temp = blockDim.x*gridDim.x;
    while(i<m*n)
    {
    	if(d_A[i] *d_B[i] <0)
    	{
    		d_B[i] = -d_B[i];
    	}

        i+=temp;
    }
     __syncthreads();
}



int main()
{
	int n =200;
	int a = n*n;
	int b = n;
	int k = n*0.1;

	float *A = new float[a*b];
	float *A_mode3 = new float[a*b];

	//gentuTensor1(A,n,n,n,k,k,k);

	for(long i = 0; i < a*b; ++i) {        
        A[i]=rand()*0.1/(RAND_MAX);
    }

	for(int i=0;i<a;i++)
    {
        for(int j = 0; j < b; ++j) {
            A_mode3[i*b+j] = A[i+j*a];          
        }
    }

	float *d_A;
	cudaMalloc((void**)&d_A,sizeof(float)*a*b);
    cudaMemcpy(d_A,A,sizeof(float)*a*b,cudaMemcpyHostToDevice);

    cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	float alpha = 1.0;
	float beta =0.0;

	float *d_ATA;
	cudaMalloc((void**)&d_ATA, sizeof(float)*b*b);

	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,b,a,&alpha,d_A,a,d_A,a,&beta,d_ATA,b);

	dim3 threads(1024,1,1);
    dim3 block0((k*k+1024-1)/1024,1,1);

    float *d_W = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    int  lwork = 0;

    cudaMalloc ((void**)&d_W, sizeof(float) * b);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnSsyevd_bufferSize(
        cusolverH,jobz,uplo,b,d_ATA,b,d_W,&lwork);
     cudaMalloc((void**)&d_work, sizeof(float)*lwork);
     cusolverDnSsyevd(
        cusolverH,jobz,uplo,b,d_ATA,b,
        d_W,d_work,lwork,devInfo);

    float *d_S,*d_ST,*d_AK,*d_U,*d_U_t;
    cudaMalloc((void**)&d_S,sizeof(float)*k*k);
    cudaMalloc((void**)&d_ST,sizeof(float)*b*k);
    cudaMalloc((void**)&d_AK,sizeof(float)*b*k); //b行k列
    cudaMalloc((void**)&d_U,sizeof(float)*a*k);
    cudaMalloc((void**)&d_U_t,sizeof(float)*a*k);

     sqrt_gpu2<<<threads,block0>>>(d_W,d_S,b,k);
     
	cublasScopy(handle,b*k,d_ATA+b*(b-k),1,d_AK,1); //后k列

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,b,k,k,&alpha,d_AK,b,d_S,k,&beta,d_ST,b);
	//cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,k,b,&alpha,d_A,a,d_ST,b,&beta,d_U,a);
	//对d_A分片乘法来做

	int slice2 = a/5;
	float *d_Amode3,*d_tempB,*d_UT;
	cudaMalloc((void**)&d_Amode3,sizeof(float)*b*slice2);
	cudaMalloc((void**)&d_tempB,sizeof(float)*k*slice2);
	cudaMalloc((void**)&d_UT,sizeof(float)*a*k);


	 for(int i = 0; i < 5; ++i) {

        cudaMemcpy(d_Amode3,A_mode3+i*b*slice2,sizeof(float)*b*slice2,cudaMemcpyHostToDevice);
        //printTensor(d_Amode3,4,4,1);
        cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k,slice2,b,
                &alpha,d_ST,b,d_Amode3,b,
                &beta,d_tempB,k
                );
                
        cublasScopy(handle,k*slice2,d_tempB,1,d_U+i*k*slice2,1);
        cudaDeviceSynchronize();
    }
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,a,k,&alpha,d_U,k,&beta,d_U,k,d_UT,a);
    dim3 block3((a*k+1024-1)/1024,1,1);
	transmission<<<1024,block3>>>(d_UT,d_U_t,a,k);
	cudaDeviceSynchronize();

	

	printTensor(d_U_t,4,4,1);



	//直接svd

	int *devInfo3 = NULL;
    float *d_work3 = NULL;
    float *d_rwork3 = NULL;
    int lwork3 = 0;
    int info_gpu3 = 0;
    float *d_S2 = NULL;
    float *d_U2 = NULL;
    float *d_VT = NULL;

    cudaMalloc ((void**)&d_S2  , sizeof(float)*b);
    cudaMalloc ((void**)&d_U2  , sizeof(float)*a*a);
    cudaMalloc ((void**)&d_VT , sizeof(float)*a*b);
    cudaMalloc ((void**)&devInfo3, sizeof(int));

    cusolverDnSgesvd_bufferSize(cusolverH,a,b,&lwork3 );
    cudaMalloc((void**)&d_work3 , sizeof(float)*lwork3);
    signed char jobu = 'S'; // all m columns of U
    signed char jobvt = 'S'; // all n columns of VT

    cusolverDnSgesvd (cusolverH,jobu,jobvt,
        a,b,d_A,a,
        d_S2,
        d_U2,
        a,  // ldu
        d_VT,
        a, // ldvt,
        d_work3,
        lwork3,
        d_rwork3,
        devInfo3);
   printTensor(d_U2,4,4,1);

   fuHao<<<1024,block3>>>(d_U2,d_U_t,a,k);

    float alpha1=-1.0;
    float re=0.0;
    float before = 0.0;

    cublasSaxpy(handle,a*k,&alpha1,d_U2,1,d_U_t,1);
    //printTensor(d_Ux2,a,b,k); 
    cublasSnrm2(handle,a*k,d_U_t,1,&re);
    cublasSnrm2(handle,a*k,d_U2,1,&before);
    cudaDeviceSynchronize();
    cout<<"error rate "<<re/before<<endl;


    // 结果，在保证了两者不存在相反数的情况下，在40000*200 的时候  误差达到了0.1

}