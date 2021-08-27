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
void QR(float *d_A,int m,int n,float *d_R,cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{
    int *devInfo2 = NULL;
    float *d_work2 = NULL;
    int  lwork_geqrf = 0;
    int  lwork_orgqr = 0;
    int  lwork2 = 0;
    float *d_tau = NULL;
    dim3 threads(1024,1,1);
    dim3 block0((n*n+1024-1)/1024,1,1);

    cudaMalloc ((void**)&d_tau, sizeof(float) * n);
    cudaMalloc ((void**)&devInfo2, sizeof(int));

    cusolverDnSgeqrf_bufferSize(cusolverH,m,n,d_A,m,&lwork_geqrf);
    cusolverDnSorgqr_bufferSize(cusolverH,m,n,n,d_A,m, d_tau,&lwork_orgqr);

    lwork2 = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
    cudaMalloc((void**)&d_work2, sizeof(float)*lwork2);

    cusolverDnSgeqrf(cusolverH,m,n,d_A,m,d_tau,d_work2,lwork2,devInfo2);

    upper<<<block0,threads>>>(d_A,d_R,m,n); // 获得R
    cudaDeviceSynchronize();
    cusolverDnSorgqr(cusolverH,m,n,n,d_A,m,d_tau,d_work2,lwork2,devInfo2); //获得 Q

    cudaFree(d_tau);
    cudaFree(devInfo2);
    cudaFree(d_work2);
}

int main()
{
	int m=1700*1700/4;
	int n= 1700;

	float *A = new float[m*n];
	for(unsigned i = 0; i < m*n; ++i) {
		A[i] = rand()*0.1/(RAND_MAX);
	}
	float *d_A,*d_R;
	cudaMalloc((void**)&d_A,sizeof(float)*m*n);
	cudaMalloc((void**)&d_R,sizeof(float)*n*n);
	cudaMemcpy(d_A,A,sizeof(float)*m*n,cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle); 	
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);

	QR(d_A,m,n,d_R,handle,cusolverH);

	printTensor(d_R,4,4,1);




}