#include "head.h"

void printTensor(dt *d_des,long m,long n,long l){
	dt *des = new dt[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(long k = 0;k<l;k++){
		for(long i  = 0;i<n;i++){
			for(long j = 0;j<m;j++){
				cout<<des[k*m*n+i*m+j]<<" ";
			}
			cout<<endl;
		}
		cout<<"~~~~~~~~~~~~~~~~"<<endl;
	}
	delete[] des;des=nullptr;
}

void printvec(dt *d_des,long m,long n,long l)
{
  dt *des = new dt[m*n*l]();
  cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for(int i = 0; i < m*n*l; ++i) {
    cout<<des[i]<<" ";
  }
  cout<<endl;
  cout<<"~~~~~~~~~~~~~~~~"<<endl;
  delete[] des;des=nullptr;
}
__global__  void floattohalf(dt *AA,half *BB,long m){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  if(i<m){
    BB[i]=__float2half(AA[i]);
    i+=temp;
  }
  __syncthreads();
}

void f2h(dt *A,half *B,long num){
  dim3 threads(512,1,1);
  dim3 blocks((num+512-1)/512,1,1); 
  floattohalf<<<blocks,threads>>>(A,B,num);
}
__global__ void mode2(dt *A,dt *B,long m,long n,long r)
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;
	while(i<m*r*n){
		long long row=i/n;
		long long col = i%n;
		long long ge = i/(m*n);
		B[i]=A[(row-ge*m)+(col*m+ge*m*n)];		
		i+=temp;
	}
	__syncthreads();	
}

__global__ void mode2h(half *A,half *B,long m,long n,long r)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long long temp = blockDim.x*gridDim.x;
  while(i<m*r*n){
    long long row=i/n;
    long long col = i%n;
    long long ge = i/(m*n);
    B[i]=A[(row-ge*m)+(col*m+ge*m*n)];    
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
__global__ void transmission(dt *d_A,dt *d_B,long a,long b)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;
    while(i<a*b)
    {
      long col=i/a+1;
      long row=i%a;
      d_B[a*(b-col)+row]=d_A[i];
      i+=temp;
    }
 __syncthreads();
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
void genHtensor(dt *X,long a,long b,long c)
{	
	srand((unsigned)time(NULL)); 
   int size=a;
   int k[5];
   int q=7;
   int w=3;
   for(int i =0;i<5;i++){
        k[i]=(rand() % (q-w+1))+ w; //3-10随机整数
       //k[i]=(int)(a*0.2); 
            
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



void gesvda(dt *d_A,dt *d_U,int a,int b,int k) //降序
{
  //A-input   U left  V  right

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
    //printTensor(d_S,6,1,1);
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
    //cout<<"svd zhong de U:"<<endl;;printTensor(d_U,5,5,1);

    cudaFree(d_S);
    //cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaStreamDestroy(stream);
    cusolverDnDestroy(cusolverH);
}

void Dngesvd(dt *d_A,dt *d_U,int a,int b)   // a > b  降序
{

    float *d_S = NULL;
    
    float *d_VT = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_rwork = NULL;
    float *d_W = NULL;  // W = S*VT

    int lwork = 0;
    int info_gpu = 0;
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    printTensor(d_A,5,5,1);

    cudaMalloc ((void**)&d_S  , sizeof(float)*b);
    cudaMalloc ((void**)&d_VT , sizeof(float)*a*b);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaMalloc ((void**)&d_W  , sizeof(float)*a*b);

    cusolverDnSgesvd_bufferSize(
        cusolverH,
        a,
        b,
        &lwork );
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'N'; // all n columns of VT
     cusolverDnSgesvd (
        cusolverH,
        jobu,
        jobvt,
        a,
        b,
        d_A,
        a,
        d_S,
        d_U,
        a,  // ldu
        d_VT,
        a, // ldvt,
        d_work,
        lwork,
        d_rwork,
        devInfo);
    cudaDeviceSynchronize();
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("after gesvd: info_gpu = %d\n", info_gpu); 

    printTensor(d_U,5,5,1);

    if (d_S    ) cudaFree(d_S);    
    if (d_VT   ) cudaFree(d_VT);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
    if (d_W    ) cudaFree(d_W);

    if (cusolverH) cusolverDnDestroy(cusolverH);
}

void gesvdj(dt *d_A,dt *d_U,int m,int n)
{

    float *d_V = NULL;
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
    cudaMalloc ((void**)&d_V , sizeof(float)*m*n);
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
        d_A,  /* m-by-n */
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
        d_A,   /* m-by-n */
        m,   /* leading dimension of A */
        d_S,   /* min(m,n)  */               /* the singular values in descending order */
        d_U,   /* m-by-m if econ = 0 */          
        m,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */               /* n-by-min(m,n) if econ = 1  */
        m,   /* leading dimension of V, ldv >= max(1,n) */
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

printTensor(d_U,4,4,1);
    if (d_S    ) cudaFree(d_S);
    if (d_V    ) cudaFree(d_V);
    if (d_info) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
}

void QR(float *d_A,int m,int n,cusolverDnHandle_t cusolverH)
{
    float *d_work = NULL, *d_tau = NULL;
    int *devInfo = NULL;
     int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    int lwork = 0;
    int info_gpu = 0;
    cudaMalloc((void**)&d_tau, sizeof(float)*n);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverDnSgeqrf_bufferSize(
        cusolverH,
        m,
        n,
        d_A,
        m,
        &lwork_geqrf);
    cusolverDnSorgqr_bufferSize(
        cusolverH,
        m,
        n,
        n,
        d_A,
        m,
        d_tau,
        &lwork_orgqr);
    lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
    cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    cusolverDnSgeqrf(
        cusolverH,
        m,
        n,
        d_A,
        m,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cusolverDnSorgqr(
        cusolverH,
        m,
        n,
        n,
        d_A,
        m,
        d_tau,
        d_work,
        lwork,
        devInfo);

    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
     printf("after geqrf: info_gpu = %d\n", info_gpu);

    if (d_work) cudaFree(d_work); d_work = NULL;
    if (devInfo) cudaFree(devInfo); devInfo = NULL;
    if (d_tau) cudaFree(d_tau); d_tau = NULL;
}
void svd(float *d_B,int m,int n,float *d_UT,float *d_S,float *d_V,cublasHandle_t cublasH,cusolverDnHandle_t cusolverH)
{
    float *d_BT = NULL, *d_U = NULL;
    float *d_work = NULL, *d_rwork = NULL;
    int *devInfo = NULL;
    int lwork = 0;

    float alpha = 1.0;
    float beta = 0.0;

    cudaMalloc((void**)&d_BT, sizeof(float)*m*n);
    cudaMalloc((void**)&d_U, sizeof(float)*m*m);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cublasSgeam(cublasH,CUBLAS_OP_T, CUBLAS_OP_N, n, m,&alpha,d_B, m,&beta,d_B, n,d_BT, n);

    cusolverDnSgesvd_bufferSize(cusolverH,n,m,&lwork );
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolverDnSgesvd(cusolverH,jobu,jobvt,
        n, m,d_BT,n,d_S,d_V,n,  // ldu
        d_U,m, // ldvt,
        d_work,lwork,d_rwork,devInfo);

    cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,  m, m,&alpha, d_U, m,&beta,d_U, m,d_UT, m);

    if(d_BT) cudaFree(d_BT);
    if(d_U) cudaFree(d_U); 
    if(d_work) cudaFree(d_work);
    if(devInfo) cudaFree(devInfo);
    if(d_rwork) cudaFree(d_rwork); 

}
void rsvd(float *d_A,int m,int n,int ks,float *d_U,cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{
    float *d_B,*d_C;
    ks = ks+8;
    float alpha = 1.0;
    float beta =0.0;

    cudaMalloc((void**)&d_B, sizeof(float)*n*ks);
    cudaMalloc((void**)&d_C,sizeof(float)*m*ks);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_B, n*ks, 0, 1);
    //2、 C = AB  m*(ks)
    //cout<<"-----1-----"<<endl;
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, 
                m, ks, n,
                &alpha,d_A,m,d_B,n,&beta,d_C,m);
    //3、 C = QR（C）
    //cout<<"QR befor"<<endl; printTensor(d_C,3,3,1);
    QR(d_C,m,ks,cusolverH);
   // cout<<"QR after"<<endl; printTensor(d_C,3,3,1);
    //4、 B = C'*A  2r * n
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N, 
                ks, n,m,
                &alpha,d_C,m,d_A,m,&beta,d_B,ks);
    // 5、B 本来就是 (2*r) * (2*r) 直接用SVD
     float *d_work = NULL;
    int lwork = 0;
    int *devInfo = NULL;
    float *d_S = NULL;
    float *d_U_svd;
    float *d_VT;
    float *d_rwork = NULL;
    cudaMalloc ((void**)&d_S  , sizeof(float)*ks);
    cudaMalloc ((void**)&devInfo, sizeof(int));    
    cudaMalloc ((void**)&d_U_svd, sizeof(float)*ks*ks);
    cudaMalloc ((void**)&d_VT , sizeof(float)*ks*ks);
    cusolverDnSgesvd_bufferSize(
        cusolverH,
        ks,
        ks,
        &lwork );
     cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolverDnSgesvd (
        cusolverH,
        jobu,
        jobvt,
        ks,
        ks,
        d_B,
        ks,
        d_S,
        d_U_svd,
        ks,  // ldu
        d_VT,
        ks, // ldvt,
        d_work,
        lwork,
        d_rwork,
        devInfo);

   // dim3 threads(1024,1,1);
   // dim3 block0((2*r*2*r+1024-1)/1024,1,1);  
    // 6、 U = C*BBT  size m*(2*r)
    float *d_U_temp;
    cudaMalloc ((void**)&d_U_temp, sizeof(float) *m* ks);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, 
                m,ks,ks,
                &alpha,d_C,m,d_U_svd,ks,&beta,d_U_temp,m);
    ks = ks-8;
    cublasScopy(handle,m*ks,d_U_temp,1,d_U,1);
    //printTensor(d_U,4,4,1); 

    cudaFree(d_S);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_U_temp);
    cudaFree(d_U_svd);
    cudaFree(d_VT);
    cudaFree(d_work);
    cudaFree(devInfo);

}

void evd(float *d_A,int m,cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{

    float *d_W;
    float *d_work = NULL;
    int  lwork = 0;
    int *devInfo = NULL;
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaMalloc((void**)&d_W,sizeof(float)*m);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    cusolverDnSsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        m,
        d_W,
        &lwork);
    cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    cusolverDnSsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        m,
        d_W,
        d_work,
        lwork,
        devInfo);
    cudaFree(d_W);
    cudaFree(devInfo);
    cudaFree(d_work);
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
    signed char jobvt = 'N';
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