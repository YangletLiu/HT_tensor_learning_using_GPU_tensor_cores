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
oid  gentuTensor1(float *X,long a,long b,long c,long r1,long r2,long r3)
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

int main()
{	
	int a = 100;
    int b = 100;
    int c = 100;
    int k = 10;

    float *A = new float[a*b*c];
    float *A_mode3 = new float[a*b*c];

    gentuTensor1(A,a,b,c,k,k,k);
    
}