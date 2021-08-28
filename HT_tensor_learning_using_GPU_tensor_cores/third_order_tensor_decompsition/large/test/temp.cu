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

__global__  void floattohalf(float *AA,half *BB,long m){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  if(i<m){
    BB[i]=__float2half(AA[i]);
    i+=temp;
  }
  __syncthreads();
}
__global__  void floattohalf2(float *AA,half *BB,long m,long n){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  if(i<m*n){
  	int row = i%m;
  	int col = i/m;

    BB[row*n+col]=__float2half(AA[col*m+row]);
    i+=temp;
  }
  __syncthreads();
}

void f2h(float *A,half *B,long num){
  dim3 threads(512,1,1);
  dim3 blocks((num+512-1)/512,1,1); 
  floattohalf<<<blocks,threads>>>(A,B,num);
}
void f2t_h(float *A,half *B,long m,long n){
  dim3 threads(512,1,1);
  dim3 blocks((m*n+512-1)/512,1,1); 
  floattohalf2<<<blocks,threads>>>(A,B,m,n);
}


int main()
{

for(int l=80;l<8001;l=l+720){
	cout<<"size :"<<l<<endl;
	   int m =4;
    int n =l;

    float *A,*d_A,*B,*d_B,*d_C;
    cudaHostAlloc((void**)&A,sizeof(float)*m*n, 0);
    cudaHostAlloc((void**)&B,sizeof(float)*m*n, 0);

    cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    cudaMalloc((void**)&d_B,sizeof(float)*m*n);
    cudaMalloc((void**)&d_C,sizeof(float)*m*m);

    for(long long  i = 0; i < m*n; ++i) {
        A[i] = i+1;
        //A[i]= rand()*0.1/(RAND_MAX*0.1);
    }
    for(long long  i = 0; i < m*n; ++i) {
        B[i] = i+1;
        //B[i]= rand()*0.1/(RAND_MAX*0.1);
    }
    cudaMemcpy(d_A,A,sizeof(float)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*n*m,cudaMemcpyHostToDevice);


    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;

    //1、CUDA core 且作为基准
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,m,n,&alpha,d_A,m,d_B,n,&beta,d_C,m);


 /*  		
    for(unsigned i = 0; i < 10; ++i) {

    	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,m,n,&alpha,d_A,m,d_B,n,&beta,d_C,m);
    	cudaDeviceSynchronize();
    }
*/
    	

    // 2、直接使用 tensor core
    half *h_A,*h_B;
    cudaMalloc((void**)&h_A,sizeof(half)*m*n);
    cudaMalloc((void**)&h_B,sizeof(half)*m*n);
    cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
     cudaEvent_t start,stop;
    float elapse__halfime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);    
    for(unsigned i = 0; i < 10; ++i) {
    	f2h(d_A,h_A,m*n);
    	f2h(d_B,h_B,m*n);
    	cublasGemmEx(handle,
    	             CUBLAS_OP_N,
    	             CUBLAS_OP_N,
    	             m,
    	             m,
    	             n,
                           &alpha,h_A,CUDA_R_16F,m,h_B,CUDA_R_16F,n,
                           &beta,d_C,CUDA_R_32F,m,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
   cudaEventRecord(stop,0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapse__halfime,start,stop);
    cout<<elapse__halfime<<endl;

   

	
/*
    // 3、分块之后再使用，其中A每块列为4，B每块行为4
    int d = n/4; // 一共 d 块
    float beta1 = 1.0;
    float *d_C2;
    cudaMalloc((void**)&d_C2,sizeof(float)*m*m);
    half *h_A,*h_BT;
    cudaMalloc((void**)&h_A,sizeof(half)*m*n);
    cudaMalloc((void**)&h_BT,sizeof(half)*m*n);
   
		f2h(d_A,h_A,m*n);
    	f2t_h(d_B,h_BT,m,n);
    //cublasSgeam(handle,CUBLAS_OP_T.CUBLAS_OP_T,m,n,&alpha,d_B,n,&beta,d_B,n,d_BT,m);
    for(unsigned i = 0; i < d; ++i) {
    	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,m,4,
    	             	   &alpha,h_A+i*m*4,CUDA_R_16F,m,h_BT+i*m*4,CUDA_R_16F,m,
                           &beta1,d_C2,CUDA_R_32F,m,CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }**/
 

		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		cublasDestroy(handle);
		cudaDeviceReset();

 }  	
}