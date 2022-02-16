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
	for(int i = 0;i<k1;i++){
		cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N,
					k2,k3,b,
					&alpha,d_U1U2+i*k2*b,k2,
					d_U3,b,
					&beta,d_B+i*k2*k3,k2
					);
	}

	/*cublasSgemmStridedBatched(handle,
	                          CUBLAS_OP_N,CUBLAS_OP_N,
	                          k2,k3,b,
	                          &alpha,d_U1U2,k2,k2*b,d_U3,b,0,
	                          &beta,d_B,k2,k2*k3,k1	                          
	                          );*/
	cudaDeviceSynchronize();
	cudaFree(d_U1U2);
}
