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

typedef float dt;

void printvec(dt *d_des,long m,long n,long l)
{
  dt *des = new dt[m*n*l]();
  cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
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
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	cudaStream_t stream = NULL;
    syevjInfo_t syevj_params = NULL;
	int n=1200;
	dt *A = new dt[n*n];
	for(unsigned i = 0; i < n*n; ++i) {
		/* code */
		A[i] = rand()*1.0/(RAND_MAX*1.0);
	}
	dt *d_A,*d_W;
	cudaMalloc((void**)&d_A,sizeof(dt)*n*n);
	cudaMalloc ((void**)&d_W, sizeof(float) * n);
	cudaMemcpy(d_A, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);

	int *devInfo = NULL;
	float *d_work = NULL;
	int  lwork = 0;
/*
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
*/
	clock_t t1,t2;
	double times=0.0;

	t1=clock();	
	const float tol = 1.e-7;
    const int max_sweeps = 15;
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);
    cusolverDnCreateSyevjInfo(&syevj_params);
    cusolverDnXsyevjSetTolerance(
        syevj_params,
        tol);
    cusolverDnXsyevjSetMaxSweeps(
        syevj_params,
        max_sweeps);
 
     cusolverDnSsyevj_bufferSize(
        cusolverH,
        jobz,
        uplo, 
        n,
        d_A,
        n,
        d_W, 
        &lwork,
        syevj_params);
    cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    cusolverDnSsyevj(
        cusolverH,
        jobz,
        uplo, 
        n,
        d_A,
        n,
        d_W, 
        d_work,
        lwork,
        devInfo,
        syevj_params);
/*
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
	float costtime;
	cudaEventElapsedTime(&costtime,start,stop);*/

	t2=clock();
		times = (double)(t2-t1)/CLOCKS_PER_SEC;
		
		cout<<"cost time :"<<times<<"s"<<endl;

    printvec(d_W,6,2,1);




}