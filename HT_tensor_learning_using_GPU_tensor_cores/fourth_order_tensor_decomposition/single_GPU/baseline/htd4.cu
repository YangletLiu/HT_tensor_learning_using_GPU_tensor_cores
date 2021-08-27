#include "head.h"
void htd4(dt *X,long a,long b,long c,long d,int *k)
{
	dt *d_X,*d_XT;
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c*d);
  cudaMalloc((void**)&d_XT,sizeof(dt)*a*b*c*d);
	cudaMemcpy(d_X,X,sizeof(dt)*a*b*c*d,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cublasHandle_t handle;
	cublasCreate(&handle);
	dt alpha = 1.0;
	dt beta = 0.0;
	dt alpha1=-1.0;
	dt re=0.0;
	dt before = 0.0;

  dim3 block0((a*b*c*d+1024-1)/1024,1,1);
  dim3 threads(1024,1,1);

	cusolverDnHandle_t cusolverH = NULL;
	//cusolverDnCreate(&cusolverH);
  //cudaStream_t stream = NULL;
  //syevjInfo_t syevj_params = NULL;

  cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,c*d,a*b,&alpha,d_X,a*b,&beta,d_X,a*b,d_XT,c*d);
  cudaDeviceSynchronize();
	// 求 mode 展开
	dt *d_X1,*d_X2,*d_X3,*d_X4;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c*d);  // 这里用流式处理加速
	cudaMalloc((void**)&d_X3,sizeof(dt)*a*b*c*d);
  cudaMalloc((void**)&d_X1,sizeof(dt)*a*b*c*d);
  cudaMalloc((void**)&d_X4,sizeof(dt)*a*b*c*d);

  tensorToMode1<<<block0,threads>>>(d_X,d_X1,a,b,c*d);//a*b*(c*d)
  cudaDeviceSynchronize();
	//mode-2  mode-3
  tensorToMode2<<<block0,threads>>>(d_X,d_X2,a,b,c*d);//b*a*(c*d)
  cudaDeviceSynchronize();
  tensorToMode2<<<block0,threads>>>(d_X,d_X3,a*b,c,d); //c*(a*b)*d
  cudaDeviceSynchronize();
  tensorToMode3<<<block0,threads>>>(d_X,d_X4,a*b,c,d);// d*(a*b)*c
  cudaDeviceSynchronize();  
    //用于 特征分解的矩阵  d_X * d_XT
    dt *d_X1_X1,*d_X2_X2,*d_X3_X3,*d_X4_X4;
    cudaMalloc((void**)&d_X1_X1,sizeof(dt)*a*a);
    cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
    cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);
    cudaMalloc((void**)&d_X4_X4,sizeof(dt)*d*d);
/*
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*c*d,
                &alpha,d_X,a,d_X,a,
                &beta,d_X1_X1,a
                );

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a*c*d,
                &alpha,d_X2,b,d_X2,b,
                &beta,d_X2_X2,b
                );
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,c,c,a*b*d,
                &alpha,d_X3,c,d_X3,c,
                &beta,d_X3_X3,c
                );
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,d,d,a*c*b,
                &alpha,d_X,a*b*c,d_X,a*b*c,
                &beta,d_X4_X4,d
                );
    cudaFree(d_X2);
    cudaFree(d_X3);*/

    dt *d_Ux7,*d_Ux6,*d_Ux5,*d_Ux4;
    cudaMalloc((void**)&d_Ux7,sizeof(dt)*d*k[6]);
    cudaMalloc((void**)&d_Ux6,sizeof(dt)*c*k[5]);
    cudaMalloc((void**)&d_Ux5,sizeof(dt)*b*k[4]);
    cudaMalloc((void**)&d_Ux4,sizeof(dt)*a*k[3]);
//====================================================================
  qr_svd_2(d_X1,d_X1_X1,a,b*c*d);
  transmission<<<block0,threads>>>(d_X1_X1,d_Ux4,a,k[3]);
  qr_svd_2(d_X2,d_X2_X2,b,a*c*d);
  transmission<<<block0,threads>>>(d_X2_X2,d_Ux5,b,k[4]);
  cudaDeviceSynchronize();
  qr_svd_2(d_X3,d_X3_X3,c,b*a*d);
  transmission<<<block0,threads>>>(d_X3_X3,d_Ux6,c,k[2]);
  cudaDeviceSynchronize();
  qr_svd_2(d_X4,d_X4_X4,d,a*c*b);
  transmission<<<block0,threads>>>(d_X4_X4,d_Ux7,d,k[6]);
  cudaDeviceSynchronize();

  //printTensor(d_Ux7,4,4,1,1);
//====================================================================

  /*eig(d_X1_X1,a,a,cusolverH);
  eig(d_X2_X2,b,b,cusolverH);
  eig(d_X3_X3,c,c,cusolverH);
  eig(d_X4_X4,d,d,cusolverH);
  //cout<<"d_X4x4 is :"<<endl;printTensor(d_X4_X4,4,4,1,1);
  cublasScopy(handle,a*k[3],d_X1_X1+a*(a-k[3]),1,d_Ux4,1); 
  cublasScopy(handle,b*k[4],d_X2_X2+b*(b-k[4]),1,d_Ux5,1);
  cublasScopy(handle,c*k[5],d_X3_X3+c*(c-k[5]),1,d_Ux6,1);
  cublasScopy(handle,d*k[6],d_X4_X4+d*(d-k[6]),1,d_Ux7,1);
  printTensor(d_Ux7,3,3,1,1);*/
//=======================用rsvd来做======================================
  dt *d_Ux3,*d_Ux2;
  cudaMalloc((void**)&d_Ux3,sizeof(dt)*c*d*k[2]);
  cudaMalloc((void**)&d_Ux2,sizeof(dt)*a*b*k[1]);
  float *d_U;
  cudaMalloc((void**)&d_U,sizeof(float)*a*b*k[1]);
  dim3 block_t((a*b*k[1]+1024-1)/1024,1,1);
  
  cusolverDnCreate(&cusolverH);

  rsvd(d_X,d_U,a*b,c*d,k[1],handle,cusolverH);
  transmission<<<block_t,threads>>>(d_U,d_Ux2,a*b,k[1]);
  //printTensor(d_Ux2,4,4,1,1);

  rsvd(d_XT,d_U,c*d,a*b,k[2],handle,cusolverH);
  transmission<<<block_t,threads>>>(d_U,d_Ux3,a*b,k[1]);
  cudaDeviceSynchronize();
  
  cudaFree(d_U);
  cudaFree(d_XT);
//====================================================================
  

    //cout<<"d_Ux7 is :"<<endl;printTensor(d_Ux7,4,4,1,1);
    // 求 B3 = U3 x1 U6T x2 U7T
    //dt *d_Ux31,*d_Ux21;
    //cudaMalloc((void**)&d_Ux31,sizeof(float)*c*d*k[2]);
    //cudaMalloc((void**)&d_Ux21,sizeof(dt)*a*b*k[1]);
    dt *d_B3,*d_B2,*d_B1;
    cudaMalloc((void**)&d_B2,sizeof(dt)*k[3]*k[4]*k[1]);
    cudaMalloc((void**)&d_B3,sizeof(dt)*k[5]*k[6]*k[2]);
    cudaMalloc((void**)&d_B1,sizeof(dt)*k[1]*k[2]*k[0]);
    //tensorToMode1<<<block0,threads>>>(d_Ux3,d_Ux31,c,d,k[2]);
    ttm(d_Ux3,d_Ux6,d_Ux7,d_B3,c,d,k[2],k[5],k[6],handle); 
    //printTensor(d_B3,4,4,1,1);
    //求 B2 = U2 x1 U4T x2 U5T
    //tensorToMode1<<<block0,threads>>>(d_Ux2,d_Ux21,a,b,k[1]);
    ttm(d_Ux2,d_Ux4,d_Ux5,d_B2,a,b,k[1],k[3],k[4],handle);
    //printTensor(d_B2,4,4,1,1);
    //求 B1 = d_X(a*b)*(c*d) x1 U2T x2 U3T
    ttm(d_X,d_Ux2,d_Ux3,d_B1,a*b,c*d,k[0],k[1],k[2],handle);
    //printTensor(d_B1,4,4,1,1);


  //recover
    dt *d_U4B2,*d_U2_r;
    cudaMalloc((void**)&d_U4B2,sizeof(dt)*a*k[4]*k[1]);
    cudaMalloc((void**)&d_U2_r,sizeof(dt)*a*b*k[1]);

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
    //printTensor(d_U2_r,4,4,1,1);  正常
    dt *d_U6B3,*d_U3_r;
    cudaMalloc((void**)&d_U6B3,sizeof(dt)*c*k[6]*k[2]);
    cudaMalloc((void**)&d_U3_r,sizeof(dt)*c*d*k[2]);

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
              c,k[6]*k[2],k[5],
              &alpha,d_Ux6,c,d_B3,k[5],
              &beta,d_U6B3,c
                );

    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                              c,d,k[6],
                              &alpha,d_U6B3,c,c*k[6],d_Ux7,d,0,
                              &beta,d_U3_r,c,c*d,k[2]
                              );

    //printTensor(d_U3_r,4,4,1,1);  这里都是 0 存疑
    dt *d_U2B1,*d_X_r;
    cudaMalloc((void**)&d_U2B1,sizeof(dt)*a*b*k[2]);
    cudaMalloc((void**)&d_X_r,sizeof(dt)*a*b*c*d);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
              a*b,k[2],k[1],
              &alpha,d_U2_r,a*b,d_B1,k[1],&beta,d_U2B1,a*b
              );

    

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                 a*b,c*d,k[2],
                 &alpha,d_U2B1,a*b,d_U3_r,c*d,
                 &beta,d_X_r,a*b
                 );
    // printTensor(d_X_r,4,4,1,1);

    cublasSaxpy(handle,a*b*c*d,&alpha1,d_X,1,d_X_r,1); 
    cudaDeviceSynchronize();

    cublasSnrm2(handle,a*b*c*d,d_X_r,1,&re);
    cublasSnrm2(handle,a*b*c*d,d_X,1,&before);
    cudaDeviceSynchronize();
    
    cout<<"error rate "<<re/before<<endl;



    cudaFree(d_X1_X1);
    cudaFree(d_X2_X2);
    cudaFree(d_X3_X3);
    cudaFree(d_X4_X4);
    cudaFree(d_X);
    cudaFree(d_Ux7);
    cudaFree(d_Ux6);
    cudaFree(d_Ux5);
    cudaFree(d_Ux4);
    cudaFree(d_Ux3);
    cudaFree(d_Ux2);
}