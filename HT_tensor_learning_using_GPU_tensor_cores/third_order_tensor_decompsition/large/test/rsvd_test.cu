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
__global__ void transmission(float *d_A,float *d_B,long a,long b)
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
void QR(float *d_A,int m,int n,cusolverDnHandle_t cusolverH)
{
     float *d_work = NULL, *d_tau = NULL;
    int *devInfo = NULL;
    int  lwork = 0; 
    int info_gpu = 0;
    cudaMalloc((void**)&d_tau, sizeof(float)*n);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cusolverDnSgeqrf_bufferSize(cusolverH, m, n, d_A, m, &lwork);
    cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    cusolverDnSgeqrf(cusolverH, m, n, d_A, m, d_tau, d_work, lwork, devInfo);
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cusolverDnSorgqr(cusolverH,m,n,n,d_A,m,d_tau, d_work,lwork,devInfo);

    if (d_work) cudaFree(d_work); d_work = NULL;
    if (devInfo) cudaFree(devInfo); devInfo = NULL;
    if (d_tau) cudaFree(d_tau); d_tau = NULL;
}
void svd(float *d_B,int m,int n,float *d_UT,float *d_S,float *d_V,cublasHandle_t cublasH,cusolverDnHandle_t cusolverH)
{
    float *d_BT = NULL, *d_U = NULL;
    float *d_work = NULL, *d_rwork = NULL;
    int *devInfo = NULL;
    int lwork = 0,  info_gpu = 0;

    float alpha = 1.0;
    float beta = 0.0;

    cudaMalloc((void**)&d_BT, sizeof(float)*m*n);
    cudaMalloc((void**)&d_U, sizeof(float)*m*m);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cublasSgeam(cublasH,CUBLAS_OP_T, CUBLAS_OP_N, n, m,&alpha,d_B, m,&beta,d_B, n,d_BT, n);

    cusolverDnSgesvd_bufferSize(cusolverH,n,m,&lwork );
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    signed char jobu = 'S'; // all m columns of U
    signed char jobvt = 'S'; // all n columns of VT
    cusolverDnSgesvd(cusolverH,jobu,jobvt,
        n, m,d_BT,n,d_S,d_V,n,  // ldu
        d_U,m, // ldvt,
        d_work,lwork,d_rwork,devInfo);

    cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,  m, m,&alpha, d_U, m,&beta,d_U, m,d_UT, m);

    if(d_BT) cudaFree(d_BT);
    if(d_U) cudaFree(d_U); 
    if(d_work) cudaFree(d_work);
    if(devInfo) cudaFree(devInfo);
    if(d_rwork) cudaFree(d_rwork); 

}
void rsvd(float *d_A,float *d_U,int m,int n,int ks,cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{
    int p=20;
    float alpha = 1.0;
    float beta =0.0;
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    float *d_B,*d_C;
    cudaMalloc((void**)&d_B, sizeof(float)*n*ks);
    cudaMalloc((void**)&d_C,sizeof(float)*m*ks);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_B, n*ks, 0, 1);

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, m, ks, n,&alpha,d_A,m,d_B,n,&beta,d_C,m);

    QR(d_C,m,ks,cusolverH);
    for(int i=0;i<p;i++)
    {
        cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n, ks, m,&alpha,d_A,m,d_C,m,&beta,d_B,n);
        QR(d_B,n,ks,cusolverH);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, ks, n,&alpha,d_A, m, d_B, n,&beta,d_C,m);
        QR(d_C,m,ks,cusolverH);
    }
    cublasSgemm(handle,CUBLAS_OP_T,  CUBLAS_OP_N,ks, n, m,&alpha,d_C, m, d_A,  m,&beta,d_B, ks);
     float *d_UT,*d_S,*d_V;
    cudaMalloc((void**)&d_UT, sizeof(float)*ks*ks);
    cudaMalloc((void**)&d_S,sizeof(float)*ks);
    cudaMalloc((void**)&d_V,sizeof(float)*n*n);

    svd(d_B,ks,n,d_UT, d_S, d_V,handle,cusolverH);
    cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,m, ks, ks,&alpha,d_C, m,d_UT, ks,&beta,d_U, m);

    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_UT);
    cudaFree(d_S);
    cudaFree(d_V);
}
int main()
{
	int m = 10000;
    int n = 10000;
    float alpha = 1.0;
    float beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);  
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    float *A;
    cudaHostAlloc((void**)&A,sizeof(float)*m*n, 0);
    printf("init data\n");
    for(long long  i = 0; i < m*n; ++i) {
        /* code */
        A[i]= rand()*0.1/(RAND_MAX*0.1);;
    }
     int ks = 1000; 
	float *d_A,*d_AAT,*d_U,*d_U2;
	cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    cudaMalloc((void**)&d_AAT,sizeof(float)*m*n);
    cudaMalloc((void**)&d_U,sizeof(float)*m*ks);
    cudaMalloc((void**)&d_U2,sizeof(float)*m*ks);
    cudaMemcpyAsync(d_A,A,sizeof(float)*m*n,cudaMemcpyHostToDevice,0);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,m,&alpha,d_A,m,&beta,d_A,m,d_AAT,n);
   

    rsvd(d_A,d_U,m,n,ks,handle,cusolverH);
    printTensor(d_U,4,4,1);
    cudaDeviceSynchronize();
    rsvd(d_AAT,d_U2,n,m,ks,handle,cusolverH);
    printTensor(d_U2,4,4,1);
    
}