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

    // X(1) = A*G(1)*(C Kronecker B)
    // A a*r1  G(1) r1*(r2*r3)  
    dt alpha = 1.0;
    dt beta =0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                a,r2*r3,r1,
                &alpha,
                d_A,a,
                d_G,r1,
                &beta,d_AG,a
                );
    //cout<<"AG"<<endl; printTensor(d_AG,3,2,1);
    // AG * B(T)  a*r2*r3   (b*r2)T

        cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_T,
                                  a,b,r2,
                                  &alpha,
                                  d_AG,a,a*r2,
                                  d_B,b,0,
                                  &beta,
                                  d_AGB,a,a*b,r3
                                  );
    
    
//cout<<"CKRB"<<endl; printTensor(d_CkrB,3,2,1);
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                a*b,c,r3,
                &alpha,
                d_AGB,a*b,
                d_C,c,
                &beta,
                d_X,a*b
                );

    cudaMemcpyAsync(X,d_X,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,  0);
    cudaDeviceSynchronize();
    //cout<<"X"<<endl; printTensor(d_X,3,2,1);

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
            d_B[i] = sqrt(d_A[b-k+row]);     
                
        }
        else
        {
            d_B[i]=0;
        }
        
        i+=temp;
    }
     __syncthreads();
}

__global__ void upper(float *A,float *R,int m,int n)
{
    long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;

     while(i<n*n)
    {   
        long row=i/n;
        long col=i%n;
        if(row>=col)    
            R[i]=A[row*m+col];
        else
            R[i]=0;
        i+=temp;        
    }
    __syncthreads();
}

int main()
{
	 int a = 100*100;
    int b = 100;
    int k = 10; //截取后15列

    float *A = new float[a*b];
    
    gentuTensor1(A,100,100,100,k,k,k);


   

	float *d_A;
	cudaMalloc((void**)&d_A,sizeof(float)*a*b);
    cudaMemcpy(d_A,A,sizeof(float)*a*b,cudaMemcpyHostToDevice);


	//A初始化为正实数的矩阵  （能否保证为正定矩阵）————> 正定矩阵的特征值为正，可以开根号得到实数奇异值

	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);


	float alpha = 1.0;
	float beta = 0.0;


	//需要 A = (n*n) * n 的U
	//1、 A.T * A 的特征向量为 V，特征值开根号为S

	float *d_ATA;
	cudaMalloc((void**)&d_ATA, sizeof(float)*b*b);


	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,b,a,&alpha,d_A,a,d_A,a,&beta,d_ATA,b);
   // printTensor(d_ATA,b,b,1);

	//d_ATA 特征分解

    dim3 threads(1024,1,1);
    dim3 block0((k*k+1024-1)/1024,1,1); //for X2

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



     float *d_S,*d_ST,*d_AT,*d_AK;
     cudaMalloc((void**)&d_S,sizeof(float)*k*k);
     cudaMalloc((void**)&d_ST,sizeof(float)*b*k);
     cudaMalloc((void**)&d_AK,sizeof(float)*b*k); //b行k列
     
     cudaMalloc((void**)&d_AT,sizeof(float)*b*a);

     //sqrt_gpu<<<512,512>>>(d_W,d_S,a,b);
     //printTensor(d_W,8,1,1);
     sqrt_gpu2<<<threads,block0>>>(d_W,d_S,b,k);
     cudaDeviceSynchronize();
    //截取eig后V的后k列
     cublasScopy(handle,b*k,d_ATA+b*(b-k),1,d_AK,1);

     //S * V.T
     // a*b  b*b

     //下一步需要对 d_ST转置，所以这里直接求转置后的 d_ST,即 V * (d_S).T = d_ST  b*k  *  k*k

     


     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,k,k,&alpha,d_AK,b,d_S,k,&beta,d_ST,b);

     float *d_SV;
     cudaMalloc((void**)&d_SV,sizeof(float)*b*k);
     cublasScopy(handle,b*k,d_ST,1,d_SV,1);
    // printTensor(d_ST,4,4,1);

     // A= U * d_ST ==> (d_ST).T  *  U.T  = A.T   (b*k) * (k*a) = (b*a)
     //此时的  d_ST已经是 (b*k), 直接求
     //首先 QR分解   (LU分解的方程求解需要 方阵)
     

     cublasSgeam( handle,CUBLAS_OP_T, CUBLAS_OP_T,b, a,&alpha,d_A, a,&beta,d_A, a,d_AT,b); //A转置
     //printTensor(d_AT,4,4,1);
    

     int *devInfo2 = NULL;
    float *d_work2 = NULL;
    int  lwork_geqrf = 0;
    int  lwork_orgqr = 0;
    int  lwork2 = 0;
    float *d_R;
    cudaMalloc((void**)&d_R,sizeof(float)*k*k);
     float *d_tau = NULL;
    cudaMalloc ((void**)&d_tau, sizeof(float) * k);

    cudaMalloc ((void**)&devInfo2, sizeof(int));
     cusolverDnSgeqrf_bufferSize(
        cusolverH,
        b,
        k,
        d_ST,
        b,
        &lwork_geqrf);
     cusolverDnSorgqr_bufferSize(cusolverH,
                                b,
                                k,
                                k,
                                d_ST,
                                b,
                                d_tau,
                                &lwork_orgqr);
     lwork2 = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
     cudaMalloc((void**)&d_work2, sizeof(float)*lwork2);

     cusolverDnSgeqrf(cusolverH,
                     b,k,
                     d_ST,b,
                     d_tau,d_work2,lwork2,devInfo2);
     upper<<<threads,block0>>>(d_ST,d_R,b,k); //将R 分离出来

      cusolverDnSorgqr(cusolverH,   // 获得Q b*k
                     b,k,k,d_ST,
                     b,
                     d_tau,d_work2,lwork2,devInfo2);
    /*
    * 这里开始需要分片来进行
    */
    //Q.T * A.T  (k*b)  * (b*a) --> (k*a)
    float *d_B;
    cudaMalloc((void**)&d_B,sizeof(float)*k*a);
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k,a,b,
                &alpha,d_ST,b,d_AT,b,
                &beta,d_B,k
                );

    cublasStrsm(  
         handle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N, 
         CUBLAS_DIAG_NON_UNIT,
         k,
         a,
         &alpha,
         d_R,
         k,
         d_B,
         k);
    printTensor(d_B,k,a,1);

    float *d_A2;
    cudaMalloc((void**)&d_A2,sizeof(float)*a*b);
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,a,b,k,
                &alpha,d_B,k,d_SV,b,
                &beta,d_A2,a
                );


    //d_r=-d_X + d_r
    float before = 0.0;
    float re = 0.0;
    float alpha1 = -1.0;
    cublasSaxpy(handle,a*b,&alpha1,d_A,1,d_A2,1); 
    cudaDeviceSynchronize();

    cublasSnrm2(handle,a*b,d_A2,1,&re);
    cublasSnrm2(handle,a*b,d_A,1,&before);


    cudaDeviceSynchronize();
    cout<<"error rate "<<re/before<<endl;


}