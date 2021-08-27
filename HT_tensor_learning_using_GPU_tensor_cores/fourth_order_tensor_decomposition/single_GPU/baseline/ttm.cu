#include "head.h"
void ttm(dt *d_U1,dt *d_U2,dt *d_U3,dt *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle)
{
//d_Ux3,d_Ux6,d_Ux7,d_B3,c,d,k[2],k[5],k[6]
// U1 a*b*k1 , U2  a*k2 , U3  b*k3
	dt alpha = 1.0;
	dt beta = 0.0;

	dt *d_U1U2;
	cudaMalloc((void**)&d_U1U2,sizeof(dt)*k2*b*k1);



	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
	            k2,b*k1,a,
	            &alpha,d_U2,a,d_U1,a,
	            &beta,d_U1U2,k2
                );

	cublasSgemmStridedBatched(handle,
	                          CUBLAS_OP_N,CUBLAS_OP_N,
	                          k2,k3,b,
	                          &alpha,d_U1U2,k2,k2*b,d_U3,b,0,
	                          &beta,d_B,k2,k2*k3,k1	                          
	                          );
	cudaDeviceSynchronize();
	cudaFree(d_U1U2);
}

void ttm_tensorcore(half *d_U1,half *d_U2,half *d_U3,dt *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle)
{

	dt alpha = 1.0;
	dt beta = 0.0;

	dt *d_U1U2;
	cudaMalloc((void**)&d_U1U2,sizeof(dt)*k2*b*k1);
	half *d_U1U2_h;
	cudaMalloc((void**)&d_U1U2_h,sizeof(half)*k2*b*k1);

	cublasGemmEx(handle,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
							k2,b*k1,a,
                            &alpha,d_U2,CUDA_R_16F,a,
                           d_U1,CUDA_R_16F,a,
                           &beta,d_U1U2,CUDA_R_32F,k2,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);

	/*cublasSgemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           k2,b*k1,a,
                           &alpha,d_U2,CUDA_R_16F,a,
                           d_U1,CUDA_R_16F,a,
                           &beta,d_U1U2,CUDA_R_32F,k2);*/
	//cout<<"TTM zhong de di yi bu :-----------"<<endl;printTensor(d_U1U2,4,4,1);
	f2h(d_U1U2,d_U1U2_h,k2*b*k1);
	cudaDeviceSynchronize();


	cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_N,
		             k2,k3,b,
		             &alpha,d_U1U2_h,CUDA_R_16F,k2,k2*b,
		             d_U3,CUDA_R_16F,b,0,
		             &beta,d_B,CUDA_R_32F,k2,k2*k3,
		             k1,
		             CUDA_R_32F,
		             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}