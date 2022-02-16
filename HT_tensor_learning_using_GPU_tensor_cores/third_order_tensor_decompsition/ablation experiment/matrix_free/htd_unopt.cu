#include "head.h"
float htd(dt *x,long a,long b,long c,int *k,dt rel_eps,dt max_rank)
{	
	
	float time_elapsed;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);       //创建Event
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);    //记录当前时间

	dt *d_X,*d_B1,*d_B2;
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);  //原 x，也是mode-1 的x
	
	cudaMemcpy(d_X,x,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	dim3 threads(1024,1,1);
	dim3 block0((a*b*c+1024-1)/1024,1,1); // mode-2



	//Node2 qr_svd
	dt *d_X_node2;
	cudaMalloc((void**)&d_X_node2,sizeof(dt)*a*b*c);
	//tensorToMode1<<<block0,threads>>>(d_X,d_X_node2,a,b,c);
	cudaMemcpy(d_X_node2,x,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	
	dt *d_X2,*d_X3,*d_X1;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_X1,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_X3,sizeof(dt)*a*b*c);
	cudaDeviceSynchronize();
	// 函数准备
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt alpha = 1.0;
	dt beta = 0.0;
	dt alpha1=-1.0;
	dt re=0.0;
	dt before = 0.0;

	dt *d_X1_X1,*d_X2_X2,*d_X3_X3;
	cudaMalloc((void**)&d_X1_X1,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);


	dt *d_Ux5,*d_Ux4,*d_Ux3,*d_Ux2;
	cudaMalloc((void**)&d_Ux5,sizeof(dt)*b*k[4]);
	cudaMalloc((void**)&d_Ux4,sizeof(dt)*a*k[3]);
	cudaMalloc((void**)&d_Ux3,sizeof(dt)*c*k[2]);
	cudaMalloc((void**)&d_Ux2,sizeof(dt)*a*b*k[1]);
	//tensorToMode1<<<block0,threads>>>(d_X,d_X1,a,b,c);
	cublasScopy(handle, a*b*c,
		d_X, 1,
		d_X1, 1);
	cudaDeviceSynchronize();
	
	tensorToMode2<<<block0,threads>>>(d_X,d_X2,a,b,c);

	cudaDeviceSynchronize();
	//tensorToMode3<<<block0,threads>>>(d_X,d_X3,a,b,c);
	cublasSgeam(handle,
		CUBLAS_OP_T, CUBLAS_OP_T,
		c, a*b,
		&alpha,
		d_X, a*b,
		&beta,
		d_X, a*b,
		d_X3, c);

	cudaDeviceSynchronize();

	/*
	*   QR-SVD
	*/
	qr_svd_2(d_X1,d_X1_X1,a,b*c);

	//svd_VT(d_X1,d_X1_X1,a,b*c,handle);
	transmission<<<block0,threads>>>(d_X1_X1,d_Ux4,a,k[3]);
	cudaDeviceSynchronize();
	

	qr_svd_2(d_X2,d_X2_X2,b,a*c);
	transmission<<<block0,threads>>>(d_X2_X2,d_Ux5,b,k[4]);
	cudaDeviceSynchronize();
	qr_svd_2(d_X3,d_X3_X3,c,b*a);
	transmission<<<block0,threads>>>(d_X3_X3,d_Ux3,c,k[2]);
	cudaDeviceSynchronize();

		
	//Node 2 non-leaf mode-(12)=mode3T, svd->ttm->B{2}

	dt *d_U;
	cudaMalloc((void**)&d_U,sizeof(dt)*a*b*c);

	cudaDeviceSynchronize();
	qr_svd(d_X_node2,d_U,a*b,c); 
	//gesvda(d_X_node2,d_U,a*b,c,20);
	transmission<<<block0,threads>>>(d_U,d_Ux2,a*b,k[1]);
	cudaDeviceSynchronize();
	// float *d_Ux2_tensor;
	// cudaMalloc((void**)&d_Ux2_tensor,sizeof(float)*a*b*k[1]);
	// tensorToMode1<<<block0,threads>>>(d_Ux2,d_Ux2_tensor,a,b,k[1]);
	//cout<<"--------"<<endl;printTensor(d_Ux2,4,4,1);            


	//U{2}还原张量后X 与 U{4} U{5} ttm
	//(1) mode-3 转置求mode-1
	
	//  ttm(U{2}的tensor X1 U{4} X2 U{5})    
	// dt *d_XU4,*d_XU2,*d_B1_2;
	// cudaMalloc((void**)&d_XU4,sizeof(dt)*k[3]*b*k[1]);
	// //cudaMalloc((void**)&d_XU4_2,sizeof(dt)*k[3]*b*k[1]);
	// cudaMalloc((void**)&d_XU2,sizeof(dt)*k[1]*c);
	// //cudaMalloc((void**)&d_XU2_2,sizeof(dt)*k[1]*c);
	 cudaMalloc((void**)&d_B1,sizeof(dt)*k[1]*k[2]); 
	// //cudaMalloc((void**)&d_B1_2,sizeof(dt)*k[1]*k[2]); 
	 cudaMalloc((void**)&d_B2,sizeof(dt)*k[3]*k[4]*k[1]);
	// //cudaMalloc((void**)&d_B2_2,sizeof(dt)*k[3]*k[4]*k[1]);
	// cudaDeviceSynchronize();
	

	ttm(d_Ux2,d_Ux4,d_Ux5,d_B2,a,b,k[1],k[3],k[4],handle);
	ttm(d_X,d_Ux2,d_Ux3,d_B1,a*b,c,k[0],k[1],k[2],handle);


	cudaDeviceSynchronize();


	cudaEventRecord(stop,0);    //记录当前时间
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
	dt *d_U4B2,*d_U2B1,*d_r,*d_U4B2_2,*d_U2B1_2,*d_r_2;
	cudaMalloc((void**)&d_U4B2,sizeof(dt)*a*k[4]*k[1]);
	cudaMalloc((void**)&d_U4B2_2,sizeof(dt)*a*k[4]*k[1]);
	cudaMalloc((void**)&d_U2B1,sizeof(dt)*a*b*k[2]);
	cudaMalloc((void**)&d_U2B1_2,sizeof(dt)*a*b*k[2]);
	cudaMalloc((void**)&d_r,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_r_2,sizeof(dt)*a*b*c);
	dt *d_U2_r,*d_U2_r_2;
	cudaMalloc((void**)&d_U2_r,sizeof(dt)*a*b*k[1]);
	cudaMalloc((void**)&d_U2_r_2,sizeof(dt)*a*b*k[1]);
	cudaDeviceSynchronize();

cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,k[4],k[3],
                          &alpha,d_Ux4,a,0,d_B2,k[3],k[3]*k[4],
                          &beta,d_U4B2_2,a,a*k[4],k[1]
                          );
tensorToMode2<<<block0,threads>>>(d_U4B2_2,d_U4B2,a,k[4],k[1]);

    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,b,a,k[4],
    						&alpha,d_Ux5,b,0,d_U4B2,k[4],k[4]*a,
    						&beta,d_U2_r_2,b,b*a,k[1]
    						); 
    tensorToMode2<<<block0,threads>>>(d_U2_r_2,d_U2_r,b,a,k[1]);					                         
    cudaDeviceSynchronize();
  
//ttm(B{1},U{2},U{3})-->U{1}  

  cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,a*b,k[2],k[1],
                            &alpha,d_U2_r,a*b,0,d_B1,k[1],k[1]*k[2],
                            &beta,d_U2B1_2,a*b,a*b*k[2],1
                            );
  tensorToMode2<<<block0,threads>>>(d_U2B1_2,d_U2B1,a*b,k[2],1);
  cudaDeviceSynchronize();
 
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                              c,a*b,k[2],
                              &alpha,d_Ux3,c,0,d_U2B1,k[2],k[2]*a*b,
                              &beta,d_r_2,c,c*a*b,1
                              );
    tensorToMode2<<<block0,threads>>>(d_r_2,d_r,c,a*b,1);
    cudaDeviceSynchronize();
    cudaFree(d_U2_r_2);
    cudaFree(d_r_2);
    cudaFree(d_U4B2_2);
    cudaFree(d_U2B1_2);
    // cout<<"recover------"<<endl;printTensor(d_r,6,3,1);
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


	cudaFree(d_X);

	cudaFree(d_Ux3);
	cudaFree(d_Ux2);
	cudaFree(d_Ux5);
	cudaFree(d_Ux4);
	cudaFree(d_B2);
	cudaFree(d_B1);

	cudaFree(d_U4B2);
	cudaFree(d_U2B1);
	cudaFree(d_X2_X2);
	cudaFree(d_X1_X1);
	cudaFree(d_X3_X3);
	cudaFree(d_U);
	cublasDestroy(handle);

	return re/before;

}