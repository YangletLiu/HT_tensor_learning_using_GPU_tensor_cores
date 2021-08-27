#include "head.h"
void htd(dt *X,long a,long b,long c,int *k)
{	
	int p,slice;  // each part process slice matrices, leave le matrix
	if(c%4==0){
		p = 4;   //4 parts
		slice = c/4; 
	}

	dim3 threads0(512,1,1);
	dim3 block00((a*a+512-1)/512,1,1); //for X2
	dim3 block1((slice*slice+512-1)/512,1,1); //for X3

	dt alpha = 1.0;
	dt beta = 0.0;
	dt beta1=1.0;
	dt alpha1=-1.0;
	dt re=0.0;
	dt before = 0.0;

	cublasHandle_t handle;
	cublasCreate(&handle); 	
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);

 	dt *d_X1_X1,*d_X2_X2,*d_X3_X3;
	cudaMalloc((void**)&d_X1_X1,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);

	dt *d_Idemat,*d_X2,*d_X3,*d_X3T,*d_Idemat3;
	cudaMalloc((void**)&d_Idemat3,sizeof(dt)*slice*slice);
	cudaMalloc((void**)&d_Idemat,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*slice);
	cudaMalloc((void**)&d_X3,sizeof(dt)*c*c);
	cudaMalloc((void**)&d_X3T,sizeof(dt)*c*slice);
	initIdeMat<<<block00,threads0>>>(d_Idemat,a);
	initIdeMat<<<block1,threads0>>>(d_Idemat3,slice);

	dt *d_Xtemp,*d_Xtemp1;
	cudaMalloc((void**)&d_Xtemp,sizeof(dt)*a*b*slice);
	cudaMalloc((void**)&d_Xtemp1,sizeof(dt)*a*b*slice);
	for(int i = 0;i<p;i++){
		cudaMemcpyAsync(d_Xtemp,X+i*a*b*slice,sizeof(dt)*a*b*slice,cudaMemcpyHostToDevice,0);
		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*slice,&alpha,d_Xtemp,a,d_Xtemp,a,&beta1,d_X1_X1,a);
		cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,a,a,&alpha,d_Xtemp,a,a*b,d_Idemat,a,0,&beta,d_X2,b,b*a,slice);
		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a,&alpha,d_X2,b,d_X2,b,&beta1,d_X2_X2,b);
		for (int j = 0;j<p;j++){
			cudaMemcpyAsync(d_Xtemp1,X+j*a*b*slice,sizeof(dt)*a*b*slice,cudaMemcpyHostToDevice,0);
			cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,slice,slice,a*b,&alpha,d_Xtemp1,a*b,d_Xtemp,a*b,&beta,d_X3+(i*p+j)*slice*slice,slice);
			cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,a*b,&alpha,d_Xtemp,1,d_Xtemp1,a*b,&beta,d_X3_X3+i*c+j,1);
		}// d_X3 is size of slice *c transpose to c*slice
		cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,slice,slice,slice,&alpha,d_X3+i*c*slice,slice,slice*slice,d_Idemat3,slice,0,&beta,d_X3T,slice,slice*slice,p);
		cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,slice,&alpha,d_X3T,slice,&beta,d_X3_X3+i*c*slice,c,d_X3_X3+i*c*slice,c);
	}		
		//cout<<"X1X1"<<endl; printTensor(d_X1_X1,4,4,1);
	//cudaFree(d_Xtemp);
	cudaFree(d_Xtemp1);
	cudaFree(d_Idemat);
	cudaFree(d_X2);
	cudaFree(d_X3);
	cudaFree(d_X3T);
	cudaFree(d_Idemat3);
	cudaDeviceSynchronize();
//==============================================================================
	cout<<"leaf node is ready"<<endl;	
	dt *d_Ux5,*d_Ux4,*d_Ux3,*d_Ux2;
	cudaMalloc((void**)&d_Ux5,sizeof(dt)*b*k[4]);
	cudaMalloc((void**)&d_Ux4,sizeof(dt)*a*k[3]);
	cudaMalloc((void**)&d_Ux3,sizeof(dt)*c*k[2]);
	cudaMalloc((void**)&d_Ux2,sizeof(dt)*a*b*k[1]);
	cudaDeviceSynchronize();
//===============这里不用是evdj的方法，也就是不用batch============================
	
	eig(d_X1_X1,a,a,cusolverH);
	eig(d_X2_X2,b,b,cusolverH);
	eig(d_X3_X3,c,c,cusolverH);

	cublasScopy(handle,a*k[3],d_X1_X1+a*(a-k[3]),1,d_Ux4,1); 
	cublasScopy(handle,b*k[4],d_X2_X2+b*(b-k[4]),1,d_Ux5,1);
	cublasScopy(handle,c*k[2],d_X3_X3+c*(c-k[2]),1,d_Ux3,1);
	printTensor(d_Ux4,4,4,1);
//============================================================================	

 
//==============================================================================
	float *d_U;
	cudaMalloc((void**)&d_U,sizeof(float)*a*b*k[1]);
	tsqr_svd(X,a,b,c,k[1],d_U,handle,cusolverH);

	dim3 threads(1024,1,1);
	dim3 block0((a*b*k[1]+1024-1)/1024,1,1);
	transmission<<<block0,threads>>>(d_U,d_Ux2,a*b,k[1]);
	cudaDeviceSynchronize();
	cudaFree(d_U);
	//cublasScopy(handle,a*b*k[1],d_Utmp+(c-k[1])*a*b,1,d_Ux2,1);
	cout<<"UX2 is :"<<endl;printTensor(d_Ux2,4,4,1);
//====================================================================
    //ttm(U{2}的tensor X1 U{4} X2 U{5})
    dt *d_B1,*d_B2;
	cudaMalloc((void**)&d_B1,sizeof(dt)*k[1]*k[2]); 
	cudaMalloc((void**)&d_B2,sizeof(dt)*k[3]*k[4]*k[1]);

	ttm(d_Ux2,d_Ux4,d_Ux5,d_B2,a,b,k[1],k[3],k[4],handle);

	float *d_u1u2;
	cudaMalloc((void**)&d_u1u2,sizeof(float)*k[1]*c);
	for(unsigned i = 0; i < p; ++i) {
		cudaMemcpyAsync(d_Xtemp,X+i*a*b*slice,sizeof(dt)*a*b*slice,cudaMemcpyHostToDevice,0);
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k[1],slice,a*b,&alpha,d_Ux2,a*b,d_Xtemp,a*b,&beta,d_u1u2+i*k[1]*slice,k[1]);		
	}
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,k[1],k[2],c,&alpha,d_u1u2,k[1],d_Ux3,c,&beta,d_B1,k[1]);
	//cout<<"B1 is :"<<endl;printTensor(d_B1,4,4,1);
	//cout<<"B2 is :"<<endl;printTensor(d_B2,4,4,1);

	cudaFree(d_X1_X1);
	cudaFree(d_X2_X2);
	cudaFree(d_X3_X3);
	cudaFree(d_B1);
	cudaFree(d_B2);
	cudaFree(d_Ux4);
	cudaFree(d_Ux3);
	cudaFree(d_Ux2);
	cudaFree(d_Ux5);
	cudaFree(d_u1u2);
	cudaFree(d_Xtemp);
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
	cudaDeviceSynchronize();
}