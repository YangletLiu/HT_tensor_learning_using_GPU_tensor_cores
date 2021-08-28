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
using namespace std;

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

	cudaSetDevice(1);
	int m = 2048;
	int n = m;
	float *A = new float[2048*2048];
	float *B = new float[2048*2048];
	for(unsigned i = 0; i < 2048*2048; ++i) {
		 A[i]=rand()*1.0/(RAND_MAX*1.0);
		 B[i]=rand()*1.0/(RAND_MAX*1.0);
	}
	float *d_A,*d_B;
	float *d_C;
	cudaMalloc((void**)&d_A,sizeof(float)*m*n);
	cudaMalloc((void**)&d_B,sizeof(float)*m*n);
	cudaMalloc((void**)&d_C,sizeof(float)*m*n);
	cudaMemcpy(d_A,A,sizeof(float)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(float)*m*n,cudaMemcpyHostToDevice);
	cublasHandle_t handle;
   cublasCreate(&handle);
   float alpha = 1.0;
   float beta = 0.0;

   cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
	            m,n,m,
	            &alpha,d_A,m,d_B,n,
	            &beta,d_C,m
	            );
    // cudaFree(d_A);
    // cudaFree(d_C);
    // cudaFree(d_B);
    //cublasDestroy(handle);

    int c;
    cin>>c;
    return 0;
    
}