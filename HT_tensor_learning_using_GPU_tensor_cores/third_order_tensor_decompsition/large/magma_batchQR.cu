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
//#include "magma_lapack.h"
#include "magma_v2.h"
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
    printTensor(d_tau,4,1,1);
    upper<<<block0,threads>>>(d_A,d_R,m,n); // 获得R
    cudaDeviceSynchronize();
    cusolverDnSorgqr(cusolverH,m,n,n,d_A,m,d_tau,d_work2,lwork2,devInfo2); //获得 Q

    cudaFree(d_tau);
    cudaFree(devInfo2);
    cudaFree(d_work2);
}
void batch_qr(float* d_t, const int m, const int n, const int batch, float* d_tau)
{
    
    if(magma_init() != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_init error!",__FUNCTION__,__LINE__);
		return;
    }
    magma_queue_t queue=NULL;
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue);
    
//	    float *h_Amagma;
// 	    float *htau_magma;
    float *d_A, *dtau_magma;
    float **dA_array = NULL;
    float **dtau_array = NULL;

    magma_int_t   *dinfo_magma;
    magma_int_t M, N, lda, ldda, min_mn;
    magma_int_t batchCount;
    magma_int_t column;

    M = m;
    N = n;
    batchCount = batch;
    min_mn = ((m<n)?m:n);
    lda    = M;
//            n2     = lda * N * batchCount;
//    ldda = ((M+31)/32)*32;
    ldda = magma_roundup( M, 32 );
//            magma_cmalloc_cpu( &h_Amagma,   n2     );
//            magma_cmalloc_cpu( &htau_magma, min_mn * batchCount );
     if(magma_smalloc( &d_A,   ldda*N * batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }

     if(magma_smalloc( &dtau_magma,  min_mn * batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }

     if(magma_imalloc( &dinfo_magma,  batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
 
     if(magma_malloc((void**) &dA_array,   batchCount * sizeof(float*) ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
     if(magma_malloc((void**) &dtau_array, batchCount * sizeof(float*) ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
     column = N * batchCount;

     magma_scopymatrix(M, column, d_t, M, d_A, ldda, queue );
//print_device_tensor(d_t,M,N,batch);	
//   magma_cprint_gpu(M*column, 1, d_t, M*column, queue );
//   magma_cprint_gpu(M, column, d_A, ldda, queue );
         
     magma_sset_pointer( dA_array, d_A, 1, 0, 0, ldda*N, batchCount, queue );
     magma_sset_pointer( dtau_array, dtau_magma, 1, 0, 0, min_mn, batchCount, queue );
  
//    magma_cprint_gpu(M, column, d_A, ldda, queue );
//    magma_cprint_gpu(M, column, d_t, M, queue );

    if( magma_sgeqrf_batched(M, N, dA_array, ldda, dtau_array, dinfo_magma, batchCount, queue) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_cgeqrf_batched!",__FUNCTION__,__LINE__);
		return;
    }

    cudaDeviceSynchronize();
//   magma_cprint_gpu(M, column, d_A, ldda, queue );
//         magma_cgetmatrix( M, column, d_A, ldda, h_Amagma, lda, queue );
//   magma_cgetmatrix(min_mn, batchCount, dtau_magma, min_mn, htau_magma, min_mn, queue );

//     magma_cgetmatrix(min_mn, batchCount, dtau_magma, min_mn, tau, min_mn, queue );
     magma_scopymatrix(min_mn, batchCount, dtau_magma, min_mn, d_tau, min_mn, queue );
     
//   magma_cprint( M, column, h_Amagma, lda);
//   magma_cprint(min_mn, batchCount, htau_magma, min_mn);

     magma_scopymatrix(M, column, d_A, ldda, d_t, lda, queue );
//print_device_tensor(d_t,M,N,batch);	
     magma_queue_destroy( queue );
     if( d_A != NULL ){ 
     magma_free( d_A   );
     d_A = NULL;
     }

     if( dtau_magma != NULL ){
     magma_free( dtau_magma  );
     dtau_magma = NULL;
     }

     if( dinfo_magma != NULL){
     magma_free( dinfo_magma );
     dinfo_magma = NULL;
     }

     if( dA_array != NULL){
     magma_free( dA_array   );
     dA_array = NULL;
     }

     if( dtau_array != NULL){
     magma_free( dtau_array  );
     dtau_array = NULL;
     }
     if( magma_finalize() != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_finalize error!",__FUNCTION__,__LINE__);
		return;
     }	
    if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

}

int main()
{
	int m = 4;
	int n = 4;
	//int batch = 2;	
	float *A = new float[m*n];
	

	float *X = new float[m*n*2];
	cublasHandle_t handle;
	cublasCreate(&handle); 
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	for(long i = 0; i < m*n*2; ++i) {
		//X[i] = rand()*0.1/(RAND_MAX);
		X[i] = i+1;
	}
	for(long i = 0; i < m*n; ++i) {
		A[i] = i+1;
	}

	float *d_A,*d_X;
	cudaMalloc((void**)&d_A,sizeof(float)*m*n);
	cudaMalloc((void**)&d_X,sizeof(float)*m*n*2);

	cudaMemcpy(d_A,A,sizeof(float)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_X,X,sizeof(float)*m*n*2,cudaMemcpyHostToDevice);

	//float *tau_array;
	//cudaMalloc((void**)&tau_array,sizeof(float)*2*n*n);
	//printTensor(d_X,4,4,2);
	float *d_R;
	cudaMalloc((void**)&d_R,sizeof(float)*n*n);
	printTensor(d_A,4,4,1);
  	//batch_qr(d_X,m, n, 2, tau_array);
  	//printTensor(tau_array,4,4,2);
  	QR(d_A,m,n,d_R,handle,cusolverH);

	//printTensor(d_X,4,4,2);
	

}