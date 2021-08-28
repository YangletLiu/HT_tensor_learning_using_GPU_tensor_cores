#include "head.h"
float htd(dt *x,long a,long b,long c,int *k)
{		

	float time_elapsed;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);       //创建Event
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);    //记录当前时间

	dt *d_X;
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);  //原 x，也是mode-1 的x	
	cudaMemcpy(d_X,x,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	//cout<<"start~~~~"<<endl;printTensor(d_X,4,4,1);


	//half *d_X_h;
	//cudaMalloc((void**)&d_X_h,sizeof(half)*a*b*c);
	//f2h(d_X,d_X_h,a*b*c);


	//Node2
	dt *d_X_node2;
	cudaMalloc((void**)&d_X_node2,sizeof(dt)*a*b*c);
	cudaMemcpy(d_X_node2,x,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);

	//half *d_X_node2_h;
	//cudaMalloc((void**)&d_X_node2_h,sizeof(half)*a*b*c);
	//f2h(d_X_node2,d_X_node2_h,a*b*c);


	dt *d_X2;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c);
	//half *d_X2_h;
	//cudaMalloc((void**)&d_X2_h,sizeof(half)*a*b*c);

	cudaDeviceSynchronize();
	// For function
	cublasHandle_t handle;
	cublasCreate(&handle);
	//cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	dt alpha = 1.0;
	dt beta = 0.0;
	dt alpha1=-1.0;
	dt re=0.0;
	dt before = 0.0;

	 cusolverDnHandle_t cusolverH = NULL;
 //  	cudaStream_t stream = NULL;
 // 	syevjInfo_t syevj_params = NULL;


	dim3 threads(1024,1,1);
	dim3 block0((a*b*c+1024-1)/1024,1,1); // for mode-2
	cudaDeviceSynchronize();
	dt *d_X1_X1,*d_X2_X2,*d_X3_X3;
	cudaMalloc((void**)&d_X1_X1,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);

	dt *d_Ux5,*d_Ux4,*d_Ux3,*d_Ux2;
	cudaMalloc((void**)&d_Ux5,sizeof(dt)*b*k[4]);
	cudaMalloc((void**)&d_Ux4,sizeof(dt)*a*k[3]);
	cudaMalloc((void**)&d_Ux3,sizeof(dt)*c*k[2]);
	cudaMalloc((void**)&d_Ux2,sizeof(dt)*a*b*k[1]);

	/*half *d_Ux5_h,*d_Ux4_h,*d_Ux3_h,*d_Ux2_h;
	cudaMalloc((void**)&d_Ux5_h,sizeof(half)*b*k[4]);
	cudaMalloc((void**)&d_Ux4_h,sizeof(half)*a*k[3]);
	cudaMalloc((void**)&d_Ux3_h,sizeof(half)*c*k[2]);
	cudaMalloc((void**)&d_Ux2_h,sizeof(half)*a*b*k[1]);*/


	cudaDeviceSynchronize();

	//1、mode-2
	mode2<<<block0,threads>>>(d_X,d_X2,a,b,c);
	
	cudaDeviceSynchronize();
	// mode-3  d_x3
	
	cublasSgemm(handle,
	            CUBLAS_OP_N,
	            CUBLAS_OP_T,
	            b,b,a*c,
	            &alpha,d_X2,b,d_X2,b,
	            &beta,d_X2_X2,b
	            );
	

	cublasSgemm(handle,
	            CUBLAS_OP_N,
	            CUBLAS_OP_T,
	            a,a,b*c,
	            &alpha,d_X,a,d_X,a,
	            &beta,d_X1_X1,a
	            );

	cublasSgemm(handle,
	            CUBLAS_OP_T,
	            CUBLAS_OP_N,
	            c,c,a*b,
	            &alpha,d_X,a*b,d_X,a*b,
	            &beta,d_X3_X3,c
	            );
	cudaDeviceSynchronize();

	cusolverDnCreate(&cusolverH);
	evd(d_X1_X1,a,handle,cusolverH);	
	cublasScopy(handle,a*k[3],d_X1_X1+a*(a-k[3]) ,1,d_Ux4,1);

	evd(d_X2_X2,b,handle,cusolverH);
	cublasScopy(handle,b*k[4],d_X2_X2+b*(b-k[4]) ,1,d_Ux5,1);
	
	evd(d_X3_X3,c,handle,cusolverH);
	cublasScopy(handle,c*k[2],d_X3_X3+c*(c-k[2]) ,1,d_Ux3,1);


	
	//d_Ux4=d_sumXXT+a*(a-k[3]);
	//d_Ux5=d_sumXXT+a*a+a*(a-k[4]);
	//d_Ux3=d_sumXXT+2*a*a+a*(a-k[2]);

	//Node 2 non-leaf mode-(12)=mode3T, svd->ttm->B{2}

	/*dt *d_U,*d_Ux2_t;
	cudaMalloc((void**)&d_U,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_Ux2_t,sizeof(dt)*a*b*c);
	gesvda(d_X_node2,d_U,a*b,c,k[1]); // 降序

	transmission<<<block0,threads>>>(d_U,d_Ux2_t,a*b,c);
	cudaDeviceSynchronize();

	//d_Ux2=d_Ux2_t+(b-k[1])*a*b;
	cublasScopy(handle,a*b*k[1],d_Ux2_t+(b-k[1])*a*b,1,d_Ux2,1);
	cudaDeviceSynchronize();*/
	dt *d_U,*d_Ux2_t;
	cudaMalloc((void**)&d_U,sizeof(dt)*a*b*c);

	// cusolverDnDestroy(cusolverH);
	// cudaStreamDestroy(stream);
	// cusolverDnDestroySyevjInfo(syevj_params);
 //    cusolverDnCreate(&cusolverH);
	//rsvd(d_X_node2,a*b,c,k[1],d_U, handle, cusolverH);
	qr_svd(d_X_node2,d_U,a*b,c);

	dim3 threads_2(1024,1,1);
	dim3 block_2((a*b*k[1]+1024-1)/1024,1,1);

	transmission<<<block_2,threads_2>>>(d_U,d_Ux2,a*b,k[1]);
	cudaDeviceSynchronize();
	//cout<<"U2 values--------"<<endl;printTensor(d_Ux2,4,4,1);

	dt *d_B2,*d_B1;
	cudaMalloc((void**)&d_B1,sizeof(dt)*k[1]*k[2]); 
	cudaMalloc((void**)&d_B2,sizeof(dt)*k[3]*k[4]*k[1]);


	//  ttm(U{2}的tensor X1 U{4} X2 U{5})
	ttm(d_Ux2,d_Ux4,d_Ux5,d_B2,a,b,k[1],k[3],k[4],handle);
	//ttm_tensorcore(d_Ux2_h,d_Ux4_h,d_Ux5_h,d_B2,a,b,k[1],k[3],k[4],handle);


	ttm(d_X,d_Ux2,d_Ux3,d_B1,a*b,c,k[0],k[1],k[2],handle);
	//ttm_tensorcore(d_X_h,d_Ux2_h,d_Ux3_h,d_B1,a*b,c,k[0],k[1],k[2],handle);
	//cout<<"tensor B1:"<<endl;;printTensor(d_B1,5,5,1);

    cudaEventRecord( stop,0);    //记录当前时间
	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
	cudaEventDestroy(start);    //destory the event
	cudaEventDestroy(stop);
	time_elapsed = time_elapsed/1000;
	cout<<"cost time :"<<time_elapsed<<"s"<<endl;




//finish decomposition B{1}->d_B1,B{2}->d_B2,U{3}->d_X3X3,U{4}->d_X1X1,U{5}->d_X2X2
//recover the tensor x
//ttm(B{2},U{4},U{5})-->U{2}
/*****************
*这里使用tensor core计算来还原原始tensor意义不大，时间上加速不多，理论上精度会有一些损失
*
*******************/
	dt *d_U4B2,*d_U2B1,*d_r;
	cudaMalloc((void**)&d_U4B2,sizeof(dt)*a*k[4]*k[1]);
	cudaMalloc((void**)&d_U2B1,sizeof(dt)*a*b*k[2]);
	cudaMalloc((void**)&d_r,sizeof(dt)*a*b*c);
	dt *d_U2_r;
	cudaMalloc((void**)&d_U2_r,sizeof(dt)*a*b*k[1]);
	cudaDeviceSynchronize();

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	            a,k[4]*k[1],k[3],
	            &alpha,d_Ux4,a,d_B2,k[3],
	            &beta,d_U4B2,a
      	      	);

    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                              a,b,k[4],
                              &alpha,d_U4B2,a,a*k[4],d_Ux5,b,0,
                              &beta,d_U2_r,a,a*b,k[1]
                              );
    cudaDeviceSynchronize();

    //printTensor(d_U2,6,6,1);
//ttm(B{1},U{2},U{3})-->U{1}  

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                a*b,k[2],k[1],
                &alpha,d_U2_r,a*b,d_B1,k[1],
                &beta,d_U2B1,a*b
                );
    /*cout<<"zhong jian d_U2_r------"<<endl;printTensor(d_U2_r,4,4,1);
    cout<<"zhong jian d_B1------"<<endl;printTensor(d_B1,4,4,1);
    cout<<"zhong jian d_U2B1------"<<endl;printTensor(d_U2B1,4,4,1);*/
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                a*b,c,k[2],
                &alpha,d_U2B1,a*b,d_Ux3,c,
                &beta,d_r,a*b
                );

    cudaDeviceSynchronize();
    //cout<<"recover------"<<endl;printTensor(d_r,6,3,1);
    //cout<<"original"<<endl;printTensor(d_X,3,3,1);
	

    
    

	
	//compute error		
	//d_r=-d_X + d_r
	cublasSaxpy(handle,a*b*c,&alpha1,d_X,1,d_r,1); 
	cudaDeviceSynchronize();

	cublasSnrm2(handle,a*b*c,d_r,1,&re);
	cublasSnrm2(handle,a*b*c,d_X,1,&before);
	cudaDeviceSynchronize();
	
	cout<<"error rate "<<re/before<<endl;
	
	ofstream fout("time.txt",ios::app);
	fout<<time_elapsed<<"  "<<re/before<<endl;
	fout.close();





	//cudaFree(d_W);
	//cudaFree(d_work);
	cudaFree(d_X);
	cudaFree(d_X2);
	cudaFree(d_X1_X1);
	cudaFree(d_X2_X2);
	cudaFree(d_X3_X3);	
	cudaFree(d_Ux3);
	cudaFree(d_Ux2);
	//cudaFree(d_XU4);
	cudaFree(d_Ux5);
	//cudaFree(d_XU4);
	//cudaFree(d_XU2);
	cudaFree(d_B2);
	cudaFree(d_B1);
	cudaFree(d_U4B2);
	cudaFree(d_U2B1);
	cudaFree(d_X2_X2);
	cudaFree(d_X1_X1);
	cudaFree(d_X3_X3);
	//cudaFree(d_U);
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
	//cudaStreamDestroy(stream);
	//cusolverDnDestroySyevjInfo(syevj_params);

	return re/before;

}