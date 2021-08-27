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
void qr_svd_2(float *d_A,float *d_U,int a,int b)  //这里 a <= b
{

     float *d_upper;    
    cudaMalloc((void**)&d_upper, sizeof(float)*a*a);

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);

  float *d_AT;
  cudaMalloc((void**)&d_AT,sizeof(float)*a*b);

  cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,b,a,
              &alpha,d_A,a,&beta,d_A,a,d_AT,b
              );

    float *TAU;
    int *devInfo=NULL;
    int lwork_geqrf = 0;

    float *d_work=NULL;
    float *d_work2=NULL;
    int lwork2 = 0;

    dim3 threads(1024,1,1);
    dim3 block0((a*b+1024-1)/1024,1,1);

    cudaMalloc((void**)&TAU, sizeof(float)*a);
    cudaMalloc ((void**)&devInfo, sizeof(int));
  cusolverDnSgeqrf_bufferSize(cusolverH,b,a,d_AT,b,&lwork_geqrf);
  cudaMalloc((void**)&d_work, sizeof(float)*lwork_geqrf);
    cusolverDnSgeqrf(cusolverH,
                     b,a,
                     d_AT,b,
                     TAU,
                     d_work,
                     lwork_geqrf,
                     devInfo
                     );
    cudaDeviceSynchronize();
    upper<<<block0,threads>>>(d_AT,d_upper,b,a); //R  a*a
    cudaDeviceSynchronize();
    cudaFree(d_AT);
    float *d_upperT;
    cudaMalloc((void**)&d_upperT,sizeof(float)*a*a);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,a,a,
                &alpha,d_upper,a,&beta,d_upper,a,d_upperT,a
                );


    float *d_W;
   cudaMalloc((void**)&d_W,sizeof(float)*a);
    float *d_RR_V;
    cudaMalloc((void**)&d_RR_V,sizeof(float)*a*a);
    //SVD
  signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'N';
    float *d_rwork=NULL;
  cusolverDnSgesvd_bufferSize(cusolverH,
                              a,a,&lwork2
                              );
  cudaMalloc((void**)&d_work2,sizeof(float)*lwork2);
  cusolverDnSgesvd (
        cusolverH,
        jobu,
        jobvt,
        a,
        a,
        d_upperT,
        a,
        d_W,
        d_U,
        a,  // ldu
        d_RR_V,
        a, // ldvt,
        d_work2,
        lwork2,
        d_rwork,
        devInfo);
  cudaDeviceSynchronize();
     cout<<"~~~~~~"<<endl;printTensor(d_U,4,4,1);
    cudaFree(d_A);
     cudaFree(d_W);
    cudaFree(TAU);
    cudaFree(d_RR_V);
    cudaFree(d_upper);
    cudaFree(d_upperT);
    cudaFree(d_work);
    cudaFree(d_work2);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(handle);
}

void gesvdj(float *d_AT,float *d_V,int b,int a)
 //需要对 d_AT做SVD，然后求出d_V
{
	int m = b,n=a;
    
    float *d_U;
   // int *devInfo = NULL;
    float *d_work = NULL;
    //float *d_rwork = NULL;
    float *d_S=NULL;
    int *d_info = NULL; 
    //float *d_W = NULL;  // W = S*VT
    int lwork = 0;
    int info = 0; 

    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
     cudaStream_t stream = NULL;
     gesvdjInfo_t gesvdj_params = NULL;
     float tol = 1.e-7;
     int max_sweeps = 15;
     cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
     cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
     cusolverDnSetStream(cusolverH, stream);
      cusolverDnCreateGesvdjInfo(&gesvdj_params);

      int econ = 1;

    cudaMalloc ((void**)&d_S  , sizeof(float)*n);
    //cudaMalloc ((void**)&d_U  , sizeof(float)*m*m);
    cudaMalloc ((void**)&d_U , sizeof(float)*m*m);
    cudaMalloc ((void**)&d_info, sizeof(int));
    //cudaMalloc ((void**)&d_W  , sizeof(float)*m*n);

   cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);

   cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);

   cusolverDnSgesvdj_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ, /* econ = 1 for economy size */
        m,    /* nubmer of rows of A, 0 <= m */
        n,    /* number of columns of A, 0 <= n  */
        d_AT,  /* m-by-n */
        m,  /* leading dimension of A */
        d_S,  /* min(m,n) */
              /* the singular values in descending order */
        d_U,  /* m-by-m if econ = 0 */
              /* m-by-min(m,n) if econ = 1 */
        m,  /* leading dimension of U, ldu >= max(1,m) */
        d_V,  /* n-by-n if econ = 0  */
              /* n-by-min(m,n) if econ = 1  */
        m,  /* leading dimension of V, ldv >= max(1,n) */
        &lwork,
        gesvdj_params);
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);

   cusolverDnSgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        m,     /* nubmer of rows of A, 0 <= m */
        n,     /* number of columns of A, 0 <= n  */
        d_AT,   /* m-by-n */
        m,   /* leading dimension of A */
        d_S,   /* min(m,n)  */               /* the singular values in descending order */
        d_U,   /* m-by-m if econ = 0 */          
        m,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */               /* n-by-min(m,n) if econ = 1  */
        n,   /* leading dimension of V, ldv >= max(1,n) */
        d_work,
        lwork,
        d_info,
        gesvdj_params);
cudaDeviceSynchronize();
cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
 if ( 0 == info ){
        printf("gesvdj converges \n");
    }else if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }else{
        printf("WARNING: info = %d : gesvdj does not converge \n", info );
    }

    if (d_S    ) cudaFree(d_S);
    if (d_V    ) cudaFree(d_V);
    if (d_info) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
}



int main()
{
	int a = 200;
	int b = 400;

	float* A = new float[a*b];
	for(long i=0;i<a*b;i++)
   {
        A[i]=rand()*1.0/(RAND_MAX*1.0);
   }
   //   用QR分解之后再SVD
   float* d_A;
   float* d_U;
   cudaMalloc((void**)&d_U,sizeof(float)*a*a);
   cudaMalloc((void**)&d_A,sizeof(float)*a*b);
   cudaMemcpy(d_A,A,sizeof(float)*a*b,cudaMemcpyHostToDevice);
   qr_svd_2(d_A,d_U,a,b);
   printTensor(d_U,3,3,1);
   // 对d_A取转置，然后取SVD之后的V，再把V转置
   cublasHandle_t handle;
   cublasCreate(&handle);
   float alpha = 1.0;
   float beta = 0.0;

   float* d_AT,*d_V,*d_VT;
   cudaMalloc((void**)&d_AT,sizeof(float)*a*b);
   cudaMalloc((void**)&d_V,sizeof(float)*a*a);
   cudaMalloc((void**)&d_VT,sizeof(float)*a*a);
   cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,b,a,&alpha,d_A,a,&beta,d_A,a,d_AT,b);
   gesvdj(d_AT,d_V,b,a);
   cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,a,a,&alpha,d_V,a,&beta,d_V,a,d_VT,a);
   printTensor(d_VT,3,3,1);

}



