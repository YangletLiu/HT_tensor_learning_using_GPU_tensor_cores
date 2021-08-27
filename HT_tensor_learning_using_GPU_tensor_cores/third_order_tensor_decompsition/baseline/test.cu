#include<iostream>
#include<fstream>
#include <assert.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include "function.h"
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda_fp16.h>
using namespace std;
__global__ void mode2(float *A,float *B,long m,long n,long r)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
  long long temp = blockDim.x*gridDim.x;
  __shared__ float temp2[8];  

  while(i<m*r*n){
    
    long long row=i/n;
    long long col = i%n;
    long long ge = i/(m*n);
    temp2[i]=A[(row-ge*m)+(col*m+ge*m*n)];
    B[i]=temp2[i]; 
    i+=temp;
  }
  __syncthreads();
}
void printvec(float *d_des,long m,long n,long l)
{
  float *des = new float[m*n*l]();
  cudaMemcpy(des,d_des,sizeof(float)*m*n*l,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for(int i = 0; i < m*n*l; ++i) {
    cout<<des[i]<<" ";
  }
  cout<<endl;
  cout<<"~~~~~~~~~~~~~~~~"<<endl;
  delete[] des;des=nullptr;
}
int main()
{
	int a=1200;
	int b=1200*1200;
	int c=1200;
  float *A = new float[a*b];
  for(long i = 0;i<a*b;i++){
    A[i] = i;   
  }
  float *d_A,*d_C;
  cudaMalloc((void**)&d_A,sizeof(float)*a*b);
  cudaMalloc((void**)&d_C,sizeof(float)*a*b);
  cudaMemcpy(d_A,A,sizeof(float)*a*b,cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;

  cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,b,a,
              &alpha,d_A,a,&beta,d_A,a,d_C,b
              );

/*	
  float *A = new float[a*b*c]();
	for(int i = 0;i<a*b*c;i++){
		A[i] = i;		
	}
	float *d_A;
	cudaMalloc((void**)&d_A,sizeof(float)*a*b*c);
	cudaMemcpy(d_A,A,sizeof(float)*a*b*c,cudaMemcpyHostToDevice);
	float *d_A2,*d_A3,*d_A1;
	cudaMalloc((void**)&d_A2,sizeof(float)*a*b*c);
	cudaMalloc((void**)&d_A1,sizeof(float)*a*b*c);
	cudaMalloc((void**)&d_A3,sizeof(float)*a*b*c);

	mode2<<<128,512>>>(d_A,d_A1,a,b,c);
	cudaDeviceSynchronize();


	printvec(d_A,a,b,c);
	printvec(d_A1,a,b,c);*/



	
}