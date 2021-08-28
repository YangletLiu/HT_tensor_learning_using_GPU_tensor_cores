#include "head.h"

void printTensor(dt *d_des,long m,long n,long l){
	dt *des = new dt[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
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
void printvec(float *d_des,long m,long n,long l)
{
  float *des = new float[m*n*l]();
  cudaMemcpy(des,d_des,sizeof(float)*m*n*l,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for(int i = 0; i < m*n*l; ++i) {
    cout<<des[i]<<" ";
  }
  cout<<endl;
  cout<<"~~~~~~~~~~~~~~~~"<<endl;
  delete[] des;des=nullptr;
}

__global__ void tensorToMode1(dt *T1,dt *T2,int m,int n,int k ){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  while(i<m*n*k){
    long tube = i/(m*n);
    long row = (i-tube*(m*n))%m;
    long col = (i-tube*(m*n))/m;
    T2[tube*m*n+col*m+row] = T1[tube*m*n+col*m+row];
    i+=temp;
  }
  __syncthreads();
  
}

__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  while(i<m*n*k){
    long tube = i/(m*n);
    long row = (i-tube*(m*n))%m;
    long col = (i-tube*(m*n))/m;
    T2[tube*m*n+row*n+col] = T1[tube*m*n+col*m+row];
    i+=temp;
  }
    __syncthreads();
}

__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  while(i<m*n*k){
    long tube = i/(m*n);
    long row = (i-tube*(m*n))%m;
    long col = (i-tube*(m*n))/m;
    T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
    i+=temp;
  }
    __syncthreads();
}

__global__ void truncate_h(dt *d_A,dt *d_B,long a,long b)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long long temp = blockDim.x*gridDim.x;
  while(i<a*b)
  {
    d_B[i]=(d_A+(a-b)*a)[i];
    i+=temp;
  }
  __syncthreads();
}
__global__ void transmission(dt *d_A,dt *d_B,long a,long b)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;
    while(i<a*b)
    {
      d_B[i]=d_A[i];
      i+=temp;
    }
 __syncthreads();
}

void genHtensor(dt *X,long a,long b,long c)
{	
	srand((unsigned)time(NULL)); 
   int size=a;
   int k[5];
   int q=7;
   int w=3;
   for(int i =0;i<5;i++){
        k[i]=(rand() % (q-w+1))+ w; //3-10随机整数 
        //k[i]=(int)(a*0.1);    
   }
   k[0]=1;
   dt *U5,*U4,*U3,*B2,*B1;
   cudaHostAlloc((void**)&U5,sizeof(dt)*size*k[4],0);
   cudaHostAlloc((void**)&U4,sizeof(dt)*size*k[3],0);
   cudaHostAlloc((void**)&U3,sizeof(dt)*size*k[2],0);
   cudaHostAlloc((void**)&B2,sizeof(dt)*k[3]*k[4]*k[1],0);
   cudaHostAlloc((void**)&B1,sizeof(dt)*k[1]*k[2]*k[0],0);

   for(long i=0;i<size*k[4];i++)
   {
        U5[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<size*k[3];i++)
   {
        U4[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<size*k[2];i++)
   {
        U3[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<k[3]*k[4]*k[1];i++)
   {
        B2[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<k[1]*k[2]*k[0];i++)
   {
        B1[i]=rand()*2.0/RAND_MAX - 1.0;
   }


   dt *d_U5,*d_U4,*d_U3,*d_B2,*d_B1;
   cudaMalloc((void**)&d_U5, sizeof(dt)*size*k[4]);
   cudaMalloc((void**)&d_U4,sizeof(dt)*size*k[3]);
   cudaMalloc((void**)&d_U3,sizeof(dt)*size*k[2]);
   cudaMalloc((void**)&d_B2,sizeof(dt)*k[3]*k[4]*k[1]);
   cudaMalloc((void**)&d_B1,sizeof(dt)*k[1]*k[2]*k[0]);

   cudaMemcpy(d_U5,U5,sizeof(dt)*size*k[4],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U4,U4,sizeof(dt)*size*k[3],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U3,U3,sizeof(dt)*size*k[2],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B2,B2,sizeof(dt)*k[3]*k[4]*k[1],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B1,B1,sizeof(dt)*k[1]*k[2]*k[0],cudaMemcpyHostToDevice);

   cublasHandle_t handle;
   cublasCreate(&handle);
   dt alpha = 1.0;
   dt beta = 0.0;

   dt *d_U2,*d_X;
   cudaMalloc((void**)&d_U2,sizeof(dt)*size*size*k[1]);
   cudaMalloc((void**)&d_X,sizeof(dt)*size*size*size);

   dt*d_U4B2;
   cudaMalloc((void**)&d_U4B2, sizeof(dt)*size*k[4]*k[1]);
   //ttm B2 x1 U4 x2 U5
   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
               size,k[4]*k[1],k[3],
               &alpha,d_U4,size,d_B2,k[3],
               &beta,d_U4B2,size
               );
   cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                             size,size,k[4],
                             &alpha,d_U4B2,size,size*k[4],d_U5,size,0,
                             &beta,d_U2,size,size*k[4],k[1]
                             );
   //ttm B1 x1 U2 x2 U3
   dt *d_U2B1;
   cudaMalloc((void**)&d_U2B1, sizeof(dt)*size*size*k[2]);
   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
               size*size,k[2],k[1],
               &alpha,d_U2,size*size,d_B1,k[1],
               &beta,d_U2B1,size*size
               );
   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
               size*size,size,k[2],
               &alpha,d_U2B1,size*size,d_U3,size,
               &beta,d_X,size*size
               );
   cudaDeviceSynchronize();
   cudaMemcpy(X,d_X,sizeof(dt)*size*size*size,cudaMemcpyDeviceToHost);
   

   cudaFreeHost(U5);
   cudaFreeHost(U4);
   cudaFreeHost(U3);
   cudaFreeHost(B2);
   cudaFreeHost(B1);

   cudaFree(d_U5);
   cudaFree(d_U4);
   cudaFree(d_U3);
   cudaFree(d_B1);
   cudaFree(d_B2);
   cudaFree(d_U4B2);
   cudaFree(d_U2B1);
   cudaFree(d_X);
   cublasDestroy(handle);
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
void qr_svd(dt *d_A,dt *d_U,int a,int b)
{

	 float *d_upper;
    
    cudaMalloc((void**)&d_upper, sizeof(float)*b*b);
   // cudaMalloc((void**)&d_U, sizeof(float)*a*b);

	   float *TAU;
    int *devInfo=NULL;
    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    int lwork = 0;
    float *d_work=NULL;
    float *d_work2=NULL;
    int lwork2 = 0;

    dim3 threads(1024,1,1);
	dim3 block0((a*b+1024-1)/1024,1,1);

    cudaMalloc((void**)&TAU, sizeof(float)*b);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0;
	float beta = 0.0;

	 cusolverDnSgeqrf_bufferSize(cusolverH,a,b,d_A,a,&lwork_geqrf);
   cusolverDnSorgqr_bufferSize(cusolverH,
  	                            a,
  	                            b,
  	                            b,
  	                            d_A,
  	                            a,
  	                            TAU,
  	                            &lwork_orgqr);
  	lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;

  	cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    cusolverDnSgeqrf(cusolverH,
                     a,b,
                     d_A,a,
                     TAU,
                     d_work,
                     lwork,
                     devInfo
                     );
    cudaDeviceSynchronize();
    upper<<<block0,threads>>>(d_A,d_upper,a,b); //R  b*b

    cudaDeviceSynchronize();

    cusolverDnSorgqr(cusolverH,   // Q a*b
                     a,b,b,d_A,
                     a,
                     TAU,
                     d_work,
                     lwork,
                     devInfo
                     );
    cudaDeviceSynchronize();   
	 float *d_W;
	 cudaMalloc((void**)&d_W,sizeof(float)*b);
	 float *d_RR;
    cudaMalloc((void**)&d_RR,sizeof(float)*b*b);
    float *d_RR_V;
    cudaMalloc((void**)&d_RR_V,sizeof(float)*b*b);
    //SVD
	signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A';
    float *d_rwork=NULL;
	cusolverDnSgesvd_bufferSize(cusolverH,
	                            b,b,&lwork2
	                            );
	cudaMalloc((void**)&d_work2,sizeof(float)*lwork2);
	cusolverDnSgesvd (
        cusolverH,
        jobu,
        jobvt,
        b,
        b,
        d_upper,
        b,
        d_W,
        d_RR,
        b,  // ldu
        d_RR_V,
        b, // ldvt,
        d_work2,
        lwork2,
        d_rwork,
        devInfo);
	cudaDeviceSynchronize();
	cudaFree(d_RR_V);

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	            a,b,b,&alpha,d_A,a,d_RR,b,&beta,d_U,a
	            );


	   cudaFree(d_A);
	   cudaFree(d_W);
    cudaFree(TAU);
    cudaFree(d_RR);
    cudaFree(d_upper);
    cudaFree(d_work);
    cudaFree(d_work2);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(handle);

}



__global__ void norm_sum(dt *A,dt *B,int a)
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;
	while(i<a)
	{
		B[a-i-1]=A[i]*A[i];
		//B[a-i-1]=A[i];
		i=i+temp;
	}
	__syncthreads();

}
__global__ void upper_1(float *R,int n)
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;

	 while(i<n*n)
	{	
		long row=i/n;
		long col=i%n;
		if(row>=col) 	
			R[i]=1;
		else
			R[i]=0;
		i+=temp;		
	}
	__syncthreads();
}
__global__ void sqrt_T(dt *A,dt *B,int a)
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;

	while(i<a)
	{
		B[a-i-1]=sqrt(A[i]);
		//B[a-i-1]=A[i];
		i+=temp;
	}
 	__syncthreads();
}

void qr_svd_2(dt *d_A,dt *d_U,int a,int b)  //这里 a <= b
{

     float *d_upper;    
    cudaMalloc((void**)&d_upper, sizeof(float)*a*a);

  cublasHandle_t handle;
  cublasCreate(&handle);
  dt alpha = 1.0;
  dt beta = 0.0;
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);

  dt *d_AT;
  cudaMalloc((void**)&d_AT,sizeof(dt)*a*b);

  cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,b,a,
              &alpha,d_A,a,&beta,d_A,a,d_AT,b
              );
    float *TAU;
    int *devInfo=NULL;
    int lwork_geqrf = 0;
    int lwork_orgqr=0;
    int lwork;
    float *d_work=NULL;
    float *d_work2=NULL;
    int lwork2 = 0;

    dim3 threads(1024,1,1);
    dim3 block0((a*b+1024-1)/1024,1,1);

    cudaMalloc((void**)&TAU, sizeof(float)*a);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cusolverDnSgeqrf_bufferSize(cusolverH,b,a,d_AT,b,&lwork_geqrf);

    cusolverDnSgeqrf_bufferSize(cusolverH,b,a,d_AT,b,&lwork_geqrf);
   cusolverDnSorgqr_bufferSize(cusolverH,
                                b,
                                a,
                                a,
                                d_AT,
                                b,
                                TAU,
                                &lwork_orgqr);
    lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;

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

    dt *d_upperT;
    cudaMalloc((void**)&d_upperT,sizeof(dt)*a*a);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,a,a,
                &alpha,d_upper,a,&beta,d_upper,a,d_upperT,a
                );


    float *d_W;
   cudaMalloc((void**)&d_W,sizeof(float)*a);
    float *d_RR_V;
    cudaMalloc((void**)&d_RR_V,sizeof(float)*a*a);
    //SVD
  signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A';
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
    
    cudaFree(d_A);
     cudaFree(d_W);
    cudaFree(TAU);
    //cudaFree(d_U);
    cudaFree(d_RR_V);
    cudaFree(d_upper);
    cudaFree(d_upperT);
    cudaFree(d_work);
    cudaFree(d_work2);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(handle);
}

void gesvda(dt *d_A,dt *d_U,int a,int b,int k)
{
  //A输入   U输出left  V输出 right

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    const int batchSize = 1;
    const long m = a;
    const int n = b;
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int rank = b;
    const long long int strideA = (long long int)lda*n;
    const long long int strideS = n;
    const long long int strideU = (long long int)ldu*n;
    const long long int strideV = (long long int)ldv*n;
    //float A[strideA*batchSize] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 10.0, 8.0, 6.0, 9.0, 7.0, 5.0};
   
    float *d_S = NULL;  /* singular values */
  
    float *d_V = NULL;  /* right singular vectors */
    int *d_info = NULL;  /* error info */
    int lwork = 0;       /* size of workspace */
    float *d_work = NULL; /* devie workspace for gesvda */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    double RnrmF[batchSize]; /* residual norm */
    int info[batchSize];  /* host copy of error info */

    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);
   
    cudaMalloc ((void**)&d_S   , sizeof(float)*strideS*batchSize);
    cudaMalloc ((void**)&d_V   , sizeof(float)*strideV*batchSize);
    cudaMalloc ((void**)&d_info, sizeof(int)*batchSize);

   

    cusolverDnSgesvdaStridedBatched_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank, /* number of singular values */
        m,    /* nubmer of rows of Aj, 0 <= m */
        n,    /* number of columns of Aj, 0 <= n  */
        d_A,     /* Aj is m-by-n */
        lda,     /* leading dimension of Aj */
        strideA, /* >= lda*n */
        d_S,     /* Sj is rank-by-1, singular values in descending order */
        strideS, /* >= rank */
        d_U,     /* Uj is m-by-rank */
        ldu,     /* leading dimension of Uj, ldu >= max(1,m) */
        strideU, /* >= ldu*rank */
        d_V,     /* Vj is n-by-rank */
        ldv,     /* leading dimension of Vj, ldv >= max(1,n) */
        strideV, /* >= ldv*rank */
        &lwork,
        batchSize /* number of matrices */
    );
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    cusolverDnSgesvdaStridedBatched(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank, /* number of singular values */
        m,    /* nubmer of rows of Aj, 0 <= m */
        n,    /* number of columns of Aj, 0 <= n  */
        d_A,     /* Aj is m-by-n */
        lda,     /* leading dimension of Aj */
        strideA, /* >= lda*n */
        d_S,     /* Sj is rank-by-1 */
                 /* the singular values in descending order */
        strideS, /* >= rank */
        d_U,     /* Uj is m-by-rank */
        ldu,     /* leading dimension of Uj, ldu >= max(1,m) */
        strideU, /* >= ldu*rank */
        d_V,     /* Vj is n-by-rank */
        ldv,     /* leading dimension of Vj, ldv >= max(1,n) */
        strideV, /* >= ldv*rank */
        d_work,
        lwork,
        d_info,
        RnrmF,
        batchSize /* number of matrices */
    );
    cudaDeviceSynchronize();
    cudaMemcpy(info, d_info, sizeof(int)*batchSize, cudaMemcpyDeviceToHost);

    if ( 0 > info[0] ){
        printf("%d-th parameter is wrong \n", -info[0]);
        exit(1);
    }
    for(int idx = 0 ; idx < batchSize; idx++){
        if ( 0 == info[idx] ){
            printf("%d-th matrix, svda converges \n", idx );
        }else{
           printf("WARNING: info[%d] = %d : svda does not converge \n", idx, info[idx] );
        }
    }

    cudaFree(d_S);
    cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaStreamDestroy(stream);
    cusolverDnDestroy(cusolverH);
}
// void gesvdj(float *d_AT,float *d_V,int b,int a)
//  //需要对 d_AT做SVD，然后求出d_V
// {
//   int m = b,n=a;
    
//     float *d_U;
//    // int *devInfo = NULL;
//     float *d_work = NULL;
//     //float *d_rwork = NULL;
//     float *d_S=NULL;
//     int *d_info = NULL; 
//     //float *d_W = NULL;  // W = S*VT
//     int lwork = 0;
//     int info = 0; 

//     cusolverDnHandle_t cusolverH;
//     cusolverDnCreate(&cusolverH);
//      cudaStream_t stream = NULL;
//      gesvdjInfo_t gesvdj_params = NULL;
//      float tol = 1.e-7;
//      int max_sweeps = 15;
//      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
//      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
//      cusolverDnSetStream(cusolverH, stream);
//       cusolverDnCreateGesvdjInfo(&gesvdj_params);

//       int econ = 1;

//     cudaMalloc ((void**)&d_S  , sizeof(float)*n);
//     //cudaMalloc ((void**)&d_U  , sizeof(float)*m*m);
//     cudaMalloc ((void**)&d_U , sizeof(float)*m*m);
//     cudaMalloc ((void**)&d_info, sizeof(int));
//     //cudaMalloc ((void**)&d_W  , sizeof(float)*m*n);

//    cusolverDnXgesvdjSetTolerance(
//         gesvdj_params,
//         tol);

//    cusolverDnXgesvdjSetMaxSweeps(
//         gesvdj_params,
//         max_sweeps);

//    cusolverDnSgesvdj_bufferSize(
//         cusolverH,
//         jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
//               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
//         econ, /* econ = 1 for economy size */
//         m,    /* nubmer of rows of A, 0 <= m */
//         n,    /* number of columns of A, 0 <= n  */
//         d_AT,  /* m-by-n */
//         m,  /* leading dimension of A */
//         d_S,  /* min(m,n) */
//               /* the singular values in descending order */
//         d_U,  /* m-by-m if econ = 0 */
//               /* m-by-min(m,n) if econ = 1 */
//         m,  /* leading dimension of U, ldu >= max(1,m) */
//         d_V,  /* n-by-n if econ = 0  */
//               /* n-by-min(m,n) if econ = 1  */
//         m,  /* leading dimension of V, ldv >= max(1,n) */
//         &lwork,
//         gesvdj_params);
//     cudaMalloc((void**)&d_work , sizeof(float)*lwork);

//    cusolverDnSgesvdj(
//         cusolverH,
//         jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
//                /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
//         econ,  /* econ = 1 for economy size */
//         m,     /* nubmer of rows of A, 0 <= m */
//         n,     /* number of columns of A, 0 <= n  */
//         d_AT,   /* m-by-n */
//         m,   /* leading dimension of A */
//         d_S,   /* min(m,n)  */               /* the singular values in descending order */
//         d_U,   /* m-by-m if econ = 0 */          
//         m,   /* leading dimension of U, ldu >= max(1,m) */
//         d_V,   /* n-by-n if econ = 0  */               /* n-by-min(m,n) if econ = 1  */
//         n,   /* leading dimension of V, ldv >= max(1,n) */
//         d_work,
//         lwork,
//         d_info,
//         gesvdj_params);
// cudaDeviceSynchronize();
// cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
//  if ( 0 == info ){
//         printf("gesvdj converges \n");
//     }else if ( 0 > info ){
//         printf("%d-th parameter is wrong \n", -info);
//         exit(1);
//     }else{
//         printf("WARNING: info = %d : gesvdj does not converge \n", info );
//     }

//     if (d_S    ) cudaFree(d_S);
//     if (d_V    ) cudaFree(d_V);
//     if (d_info) cudaFree(d_info);
//     if (d_work ) cudaFree(d_work);

//     if (cusolverH) cusolverDnDestroy(cusolverH);
//     if (stream      ) cudaStreamDestroy(stream);
//     if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
// }

// void svd_VT(float *d_A,float *d_VT,int a,int b,cublasHandle_t handle)
// {
//    //cublasHandle_t handle;
//    //cublasCreate(&handle);
//    float alpha = 1.0;
//    float beta = 0.0;

//    float* d_AT,*d_V;
//    cudaMalloc((void**)&d_AT,sizeof(float)*a*b);
//    cudaMalloc((void**)&d_V,sizeof(float)*a*a);
//    //cudaMalloc((void**)&d_VT,sizeof(float)*a*a);
//    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,b,a,&alpha,d_A,a,&beta,d_A,a,d_AT,b);
//    gesvdj(d_AT,d_V,b,a);
//    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,a,a,&alpha,d_V,a,&beta,d_V,a,d_VT,a);
//    printTensor(d_VT,3,3,1);

//    cudaFree(d_AT);
//    cudaFree(d_V);
// }

