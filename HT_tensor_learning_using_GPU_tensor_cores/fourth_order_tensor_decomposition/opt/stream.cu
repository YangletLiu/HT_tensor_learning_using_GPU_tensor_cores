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

void printTensor(float *d_des,long m,long n,long l,long k){
  float *des = new float[m*n*l*k]();
  cudaMemcpy(des,d_des,sizeof(float)*m*n*l*k,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for(int d=0;d<k;d++){
  for(int c = 0;c<l;c++){
    for(int b = 0;b<n;b++){
      for(int a = 0;a<m;a++){
        cout<<des[d*m*n*l+c*m*n+b*m+a]<<" ";
      }
      cout<<endl;
    }
    cout<<"~~~~~~~~~~~~~~~~~~~~~"<<endl;
  }
}
}

int main()
{
	float *A;
	float *B;
	cudaHostAlloc((void**)&A,sizeof(float)*16,0);
	cudaHostAlloc((void**)&B,sizeof(float)*16,0);
	for(unsigned i = 0; i < 16; ++i) {
		/* code */
		A[i] = i+1;
		B[i] = i+2;
	}
	float *d_A,*d_B,*d_C,*d_D;
	cudaMalloc((void**)&d_A,sizeof(float)*16);
	cudaMalloc((void**)&d_B,sizeof(float)*16);
	cudaMalloc((void**)&d_C,sizeof(float)*16);
	cudaMalloc((void**)&d_D,sizeof(float)*16);		
	cudaStream_t stream[2];
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha=1.0f;                                             
	float beta=0.0f;


	cudaMemcpyAsync(d_A,A, sizeof(float)*16,cudaMemcpyHostToDevice,0);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&alpha,d_A,4,d_A,4,&beta,d_C,4);

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
for(int j=0;j<10;j++){
	cudaStreamCreate(&stream[0]);	
	cublasSetStream(handle,stream[0]);
	cudaMemcpyAsync(d_A,A, sizeof(float)*16,cudaMemcpyHostToDevice,stream[0]);
	//cudaMemcpyAsync(d_A,A, sizeof(float)*16,cudaMemcpyHostToDevice,0);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&alpha,d_A,4,d_A,4,&beta,d_C,4);
	printTensor(d_C,4,4,1,1);
	cublasDestroy(handle);

	cudaStreamCreate(&stream[1]);
	//cublasCreate(&handle);
	cublasSetStream(handle,stream[1]);
	cudaMemcpyAsync(d_B,B, sizeof(float)*16,cudaMemcpyHostToDevice,stream[1]);
	//cudaMemcpyAsync(d_B,B, sizeof(float)*16,cudaMemcpyHostToDevice,0);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&alpha,d_B,4,d_B,4,&beta,d_D,4);
	printTensor(d_D,4,4,1,1);
	//cublasDestroy(handle);
	cudaStreamSynchronize(0); 
	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	std::cout << "COST TIME : " << elapsedTime/10 << std::endl;

}