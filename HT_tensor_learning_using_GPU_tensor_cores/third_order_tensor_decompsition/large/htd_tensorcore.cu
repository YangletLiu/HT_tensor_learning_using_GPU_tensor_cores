#include "head.h"
void htd(dt *X,long a,long b,long c,int *k)
{


	float time_elapsed;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);       //创建Event
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);    //记录当前时间



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
	cublasHandle_t handle;
	cublasCreate(&handle); 	
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);

	dt *d_X1_X1,*d_X2_X2,*d_X3_X3;
	cudaMalloc((void**)&d_X1_X1,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);

	dt *d_X2,*d_X3,*d_X3T,*d_Idemat3,*d_Idemat;	
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*slice);
	cudaMalloc((void**)&d_Idemat,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X3,sizeof(dt)*c*c);
	cudaMalloc((void**)&d_X3T,sizeof(dt)*c*slice);
	cudaMalloc((void**)&d_Idemat3,sizeof(dt)*slice*slice);
	initIdeMat<<<block1,threads0>>>(d_Idemat3,slice);
	initIdeMat<<<block00,threads0>>>(d_Idemat,a);


	half *h_Idemat,*h_X2;
	cudaMalloc((void**)&h_Idemat,sizeof(half)*a*a);
	cudaMalloc((void**)&h_X2,sizeof(half)*a*b*slice);
	f2h(d_Idemat,h_Idemat,a*a);

	dt *d_Xtemp,*d_Xtemp1;
	cudaMalloc((void**)&d_Xtemp,sizeof(dt)*a*b*slice);
	cudaMalloc((void**)&d_Xtemp1,sizeof(dt)*a*b*slice);
	half *h_Xtemp,*h_Xtemp1;
	cudaMalloc((void**)&h_Xtemp,sizeof(half)*a*b*slice);
	cudaMalloc((void**)&h_Xtemp1,sizeof(half)*a*b*slice);

	for(int i = 0;i<p;i++){
		cudaMemcpyAsync(d_Xtemp,X+i*a*b*slice,sizeof(dt)*a*b*slice,cudaMemcpyHostToDevice,0);
		f2h(d_Xtemp,h_Xtemp,a*b*slice);
		//cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*slice,&alpha,d_Xtemp,a,d_Xtemp,a,&beta1,d_X1_X1,a);
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           a,a,b*slice,
                           &alpha,h_Xtemp,CUDA_R_16F,a,
                           h_Xtemp,CUDA_R_16F,a,
                           &beta1,d_X1_X1,CUDA_R_32F,a,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		
		//cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,a,a,&alpha,d_Xtemp,a,a*b,d_Idemat,a,0,&beta,d_X2,b,b*a,slice);
		cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T,CUBLAS_OP_N,
		             b,a,a,
		             &alpha,h_Xtemp,CUDA_R_16F,a,a*b,
		             h_Idemat,CUDA_R_16F,a,0,
		             &beta,d_X2,CUDA_R_32F,b,a*b,slice,		             
		             CUDA_R_32F,
		             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		f2h(d_X2,h_X2,a*b*slice);
		//cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a,&alpha,d_X2,b,d_X2,b,&beta1,d_X2_X2,b);
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           b,b,a,
                           &alpha,h_X2,CUDA_R_16F,b,
                           h_X2,CUDA_R_16F,b,
                           &beta1,d_X2_X2,CUDA_R_32F,b,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		//cout<<"X2"<<endl;printTensor(d_X2_X2,3,3,1);
		for (int j = 0;j<p;j++){
			cudaMemcpyAsync(d_Xtemp1,X+j*a*b*slice,sizeof(dt)*a*b*slice,cudaMemcpyHostToDevice,0);
			//printTensor(d_Xtemp1,3,3,1);
			f2h(d_Xtemp1,h_Xtemp1,a*b*slice);
			//cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,slice,slice,a*b,&alpha,d_Xtemp1,a*b,d_Xtemp,a*b,&beta,d_X3+(i*p+j)*slice*slice,slice);
			cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           slice,slice,a*b,
                           &alpha,h_Xtemp1,CUDA_R_16F,a*b,
                           h_Xtemp,CUDA_R_16F,a*b,
                           &beta,d_X3+(i*p+j)*slice*slice,CUDA_R_32F,slice,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
			//cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,a*b,&alpha,d_Xtemp,1,d_Xtemp1,a*b,&beta,d_X3_X3+i*c+j,1);
			cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                           1,1,a*b,
                           &alpha,h_Xtemp,CUDA_R_16F,1,
                           h_Xtemp1,CUDA_R_16F,a*b,
                           &beta,d_X3_X3+i*c+j,CUDA_R_32F,1,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		}// d_X3 is size of slice *c transpose to c*slice
		cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,slice,slice,slice,&alpha,d_X3+i*c*slice,slice,slice*slice,d_Idemat3,slice,0,&beta,d_X3T,slice,slice*slice,p);
		cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,slice,&alpha,d_X3T,slice,&beta,d_X3_X3+i*c*slice,c,d_X3_X3+i*c*slice,c);
	}

	//cout<<"d_X1_X1 is :"<<endl;printTensor(d_X1_X1,4,4,1);
	//cout<<"d_X2_X2 is :"<<endl;printTensor(d_X2_X2,4,4,1);
	//cout<<"d_X3_X3 is :"<<endl;printTensor(d_X3_X3,4,4,1);

	cudaFree(d_Xtemp1);
	cudaFree(d_X2);
	cudaFree(d_X3);
	cudaFree(d_X3T);
	cudaFree(d_Idemat3);
	cudaFree(d_Idemat);

	cudaFree(h_Xtemp1);
	cudaFree(h_Idemat);
	cudaFree(h_X2);
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
	//printTensor(d_Ux4,4,4,1);
//=============================================================================	
	
	float *d_U;
	cudaMalloc((void**)&d_U,sizeof(float)*a*b*k[1]);
	cudaDeviceSynchronize();
	tsqr_svd_half(X,a,b,c,k[1],d_U,handle,cusolverH);

	dim3 threads(1024,1,1);
	dim3 block0((a*b*k[1]+1024-1)/1024,1,1);
	transmission<<<block0,threads>>>(d_U,d_Ux2,a*b,k[1]);
	cudaDeviceSynchronize();
	cudaFree(d_U);
	//printTensor(d_Ux2,3,3,1);
//====================================================================
	dt *d_B2,*d_B1;
	cudaMalloc((void**)&d_B1,sizeof(dt)*k[1]*k[2]); 
	cudaMalloc((void**)&d_B2,sizeof(dt)*k[3]*k[4]*k[1]);

	half *d_Ux5_h,*d_Ux4_h,*d_Ux3_h,*d_Ux2_h;
	cudaMalloc((void**)&d_Ux5_h,sizeof(half)*b*k[4]);
	cudaMalloc((void**)&d_Ux4_h,sizeof(half)*a*k[3]);
	cudaMalloc((void**)&d_Ux3_h,sizeof(half)*c*k[2]);
	cudaMalloc((void**)&d_Ux2_h,sizeof(half)*a*b*k[1]);
	f2h(d_Ux2,d_Ux2_h,a*b*k[1]);
	f2h(d_Ux3,d_Ux3_h,c*k[2]);
	f2h(d_Ux4,d_Ux4_h,a*k[3]);
	f2h(d_Ux5,d_Ux5_h,b*k[4]);

	ttm_tensorcore(d_Ux2_h,d_Ux4_h,d_Ux5_h,d_B2,a,b,k[1],k[3],k[4],handle);

	float *d_u1u2;
	cudaMalloc((void**)&d_u1u2,sizeof(float)*k[1]*c);
	for(unsigned i = 0; i < p; ++i) {
		cudaMemcpyAsync(d_Xtemp,X+i*a*b*slice,sizeof(dt)*a*b*slice,cudaMemcpyHostToDevice,0);
		f2h(d_Xtemp,h_Xtemp,a*b*slice);
		//cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k[1],slice,a*b,&alpha,d_Ux2,a*b,d_Xtemp,a*b,&beta,d_u1u2+i*k[1]*slice,k[1]);		
		cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           k[1],slice,a*b,
                           &alpha,d_Ux2_h,CUDA_R_16F,a*b,
                           h_Xtemp,CUDA_R_16F,a*b,
                           &beta,d_u1u2+i*k[1]*slice,CUDA_R_32F,k[1],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	}
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,k[1],k[2],c,&alpha,d_u1u2,k[1],d_Ux3,c,&beta,d_B1,k[1]);

	//cout<<"B1 is :"<<endl;printTensor(d_B1,4,4,1);
	cudaEventRecord( stop,0);    //记录当前时间
	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
	cudaEventDestroy(start);    //destory the event
	cudaEventDestroy(stop);
	time_elapsed = time_elapsed/1000;
	cout<<"cost time :"<<time_elapsed<<"s"<<endl;




	cudaFree(d_Ux2_h);
	cudaFree(d_Ux3_h);
	cudaFree(d_Ux4_h);
	cudaFree(d_Ux5_h);
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
	cudaFree(h_Xtemp);
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
}