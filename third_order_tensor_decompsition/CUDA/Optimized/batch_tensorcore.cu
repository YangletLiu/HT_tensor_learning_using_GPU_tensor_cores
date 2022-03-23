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

__global__  void floattohalf(float *AA,half *BB,long m){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  if(i<m){
    BB[i]=__float2half(AA[i]);
    i+=temp;
  }
  __syncthreads();
}

void f2h(float *A,half *B,long num){
  dim3 threads(1024,1,1);
  dim3 blocks((num+1024-1)/1024,1,1); 
  floattohalf<<<blocks,threads>>>(A,B,num);
}
void printTensor(float *d_des,long m,long n,long l){
	float *des = new float[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(float)*m*n*l,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(long k = 0;k<l;k++){
		for(long i = 0;i<n;i++){
			for(long j = 0;j<m;j++){
				cout<<des[k*m*n+i*m+j]<<" ";
			}
			cout<<endl;
		}
		cout<<"~~~~~~~~~~~~~~~~"<<endl;
	}
	delete[] des;des=nullptr;
}

int main()
{
  int a = 64;
  int b = 64;
  int c = 3;
	float *A = new float[a*b*c];
	float *B = new float[a*b*c];
    for(long i =0;i<a*b*c;i++){
       A[i] =rand()*1.0/(RAND_MAX*1.0);            
   }
   	for(long i =0;i<a*b*c;i++){
       B[i] =rand()*1.0/(RAND_MAX*1.0);            
   }

   float *d_A,*d_B,*d_C;
   cudaMalloc((void**)&d_A,sizeof(float)*a*b*c);
   cudaMalloc((void**)&d_B,sizeof(float)*a*b*c);
   cudaMalloc((void**)&d_C,sizeof(float)*a*b*c);
   cudaMemcpy(d_A,A,sizeof(float)*a*b*c,cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,B,sizeof(float)*a*b*c,cudaMemcpyHostToDevice);

   half *h_A,*h_B;
   cudaMalloc((void**)&h_A,sizeof(half)*a*b*c);
   cudaMalloc((void**)&h_B,sizeof(half)*a*b*c);

   f2h(d_A,h_A,a*b*c);
   f2h(d_B,h_B,a*b*c);

   cublasHandle_t handle;
   cublasCreate(&handle);
   cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
   float alpha = 1.0;
   float beta = 0.0;
   printTensor(d_A,3,3,1);
   cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_N,  ////这里出问题
		             a,b,b,
		             &alpha,h_A,CUDA_R_16F,a,a*b,
		             h_B,CUDA_R_16F,a,a*b,
		             &beta,d_C,CUDA_R_32F,a,a*b,c,
		             CUDA_R_32F,
		             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

   printTensor(d_C,4,4,1);
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
   cudaFree(h_A);
   cudaFree(h_B);

}