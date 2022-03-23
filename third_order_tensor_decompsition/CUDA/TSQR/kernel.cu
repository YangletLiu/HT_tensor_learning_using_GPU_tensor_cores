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

__global__ void initIdeMat(dt *AA,int m){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*m){
    int row = i%m;
    int col = i/m;
    if(row==col){
      AA[col*m+row] = 1;
    }else{
      AA[col*m+row] = 0;
    }
    i+=temp;
  }
  __syncthreads();
}

__global__ void initIdeMat_h(half *AA,int m){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*m){
    int row = i%m;
    int col = i/m;
    if(row==col){
      AA[col*m+row] = 1;
    }else{
      AA[col*m+row] = 0;
    }
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
void  gentuTensor(float *X,long a,long b,long c,long r1,long r2,long r3)
{
    float *A,*B,*C,*G;
    cudaHostAlloc((void**)&A,sizeof(float)*a*r1,0);
    cudaHostAlloc((void**)&B, sizeof(float)*b*r2,0);
    cudaHostAlloc((void**)&C, sizeof(float)*c*r3,0);
    cudaHostAlloc((void**)&G, sizeof(float)*r1*r2*r3,0);
    srand(102);

    for(long long i=0;i<a*r1;i++)
    {
        A[i] = rand()*0.1/(RAND_MAX*0.1);
    }
    for(long long i=0;i<b*r2;i++)
    {
        B[i] = rand()*0.1/(RAND_MAX*0.1);
    }
    for(long long i=0;i<c*r3;i++)
    {
        C[i] = rand()*0.1/(RAND_MAX*0.1);
    }
    for(long long i=0;i<r1*r2*r3;i++)
    {
        G[i] = rand()*0.1/(RAND_MAX*0.1);
    }
    float * d_A,*d_B,*d_C,*d_X,*d_G;
    cudaMalloc((void**)&d_A,sizeof(float)*a*r1);
    cudaMalloc((void**)&d_B,sizeof(float)*b*r2);
    cudaMalloc((void**)&d_C,sizeof(float)*c*r3);
    cudaMalloc((void**)&d_X,sizeof(float)*a*b*c);
    cudaMalloc((void**)&d_G,sizeof(float)*r2*r1*r3);

    cudaMemcpyAsync(d_A, A,sizeof(float)*a*r1,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_B, B,sizeof(float)*b*r2,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_C, C,sizeof(float)*c*r3,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_G, G,sizeof(float)*r1*r2*r3,cudaMemcpyHostToDevice,0);
    float *d_AG,*d_AGB;
    cudaMalloc((void**)&d_AG,sizeof(float)*a*r2*r3);
    cudaMalloc((void**)&d_AGB,sizeof(float)*b*a*r3);

    float alpha = 1.0;
    float beta =0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3,r1, &alpha,d_A,a,d_G,r1,&beta,d_AG,a);
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,
                                  &alpha,d_AG,a,a*r2,d_B,b,0,
                                  &beta,d_AGB,a,a*b,r3);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,
                &alpha,d_AGB,a*b,d_C,c,&beta,d_X,a*b);

    cudaMemcpyAsync(X,d_X,sizeof(float)*a*b*c,cudaMemcpyDeviceToHost,  0);

    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_G);
    cudaFree(d_AGB);
    cudaFree(d_AG);
    cudaFree(d_X);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(G);
    cublasDestroy(handle);
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
        U5[i]=rand()*1.0/(RAND_MAX*1.0);
   }
   for(long i=0;i<size*k[3];i++)
   {
        U4[i]=rand()*1.0/(RAND_MAX*1.0);
   }
   for(long i=0;i<size*k[2];i++)
   {
        U3[i]=rand()*1.0/(RAND_MAX*1.0);
   }
   for(long i=0;i<k[3]*k[4]*k[1];i++)
   {
        B2[i]=rand()*1.0/(RAND_MAX*1.0);
   }
   for(long i=0;i<k[1]*k[2]*k[0];i++)
   {
        B1[i]=rand()*1.0/(RAND_MAX*1.0);
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
   //printTensor(d_X,4,4,1); 生成的d_X 是正确的

   cudaMemcpy(X,d_X,sizeof(dt)*size*size*size,cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();

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

void eig(float *d_A,int m,int n,cusolverDnHandle_t cusolverH)
{
    float *d_W = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    int  lwork = 0;

    cudaMalloc ((void**)&d_W, sizeof(float) * m);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnSsyevd_bufferSize(cusolverH,jobz,uplo,m,d_A,m,d_W,&lwork);
    cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    cusolverDnSsyevd(cusolverH,jobz,uplo,m,d_A,m,d_W,d_work,lwork,devInfo);
    cudaDeviceSynchronize();
   // cout<<"in the function :"<<endl;printTensor(d_A,4,4,1);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);

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

    upper<<<block0,threads>>>(d_A,d_R,m,n); // 获得R
    cudaDeviceSynchronize();
    cusolverDnSorgqr(cusolverH,m,n,n,d_A,m,d_tau,d_work2,lwork2,devInfo2); //获得 Q

    cudaFree(d_tau);
    cudaFree(devInfo2);
    cudaFree(d_work2);
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
    cudaFree(d_V);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaStreamDestroy(stream);
    cusolverDnDestroy(cusolverH);
}

void svd(float *d_A,float *d_Rk,int a,int b,int k,cublasHandle_t handle,cusolverDnHandle_t cusolverH)   // a > b  降序
{
    float *d_U = NULL;
    float *d_S = NULL;   
    float *d_VT = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_rwork = NULL;
    float *d_W = NULL;  // W = S*VT

    int lwork = 0;
    int info_gpu = 0;
    cudaMalloc ((void**)&d_U,sizeof(float)*a*a);
    cudaMalloc ((void**)&d_S  , sizeof(float)*b);
    cudaMalloc ((void**)&d_VT , sizeof(float)*a*b);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaMalloc ((void**)&d_W  , sizeof(float)*a*b);

    cusolverDnSgesvd_bufferSize(cusolverH,a,b,&lwork );
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    signed char jobu = 'S'; // all m columns of U
    signed char jobvt = 'S'; // all n columns of VT
     cusolverDnSgesvd (
        cusolverH,jobu,jobvt,a,b,d_A,a,d_S,d_U,a,  // ldu
        d_VT,
        a, // ldvt,
        d_work,lwork,d_rwork,devInfo);
    cudaDeviceSynchronize();
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("after gesvd: info_gpu = %d\n", info_gpu); 

    cublasScopy(handle,a*k,d_U,1,d_Rk,1);

    if (d_S    ) cudaFree(d_S);    
    if (d_VT   ) cudaFree(d_VT);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
}
void tsqr_svd(float *A,int a,int b,int c,int k,float *d_U,cublasHandle_t handle, cusolverDnHandle_t cusolverH)
{
    long m = a*b;
    int n=c;

    int p=4;//将A按照垂直方向分为p块，每块size (m/4)*n
    int d = m/4;

    float *B[4];
    cudaHostAlloc((void**)&B[0],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[1],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[2],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[3],sizeof(float)*d*n,0);
    long long row,col;
    for(long long i = 0; i < m*n; ++i) {
      row = i%m;
      col = i/m;
      if(row/d <1)
      {
        B[0][row+col*d] = A[i];
      }else if(row/d<2 && row/d>=1)
      {
        B[1][row-d+col*d] = A[i];
      }else if(row/d<3 && row/d>=2)
      {
        B[2][row-2*d+col*d] = A[i];
      }else if(row/d<4 && row/d>=3)
      {
        B[3][row-3*d+col*d] = A[i];
      }     
    }


    float alpha = 1.0;
    float beta = 0.0;
    float *d_B;
    cudaMalloc((void**)&d_B,sizeof(float)*d*n);

    float *d_R[4];
   // printTensor(d_B[0],d,n,1);
    for(int i =0;i<p;i++)
    {
      cudaMalloc((void**)&d_R[i],sizeof(float)*n*n); 
      cudaMemcpy(d_B,B[i],sizeof(float)*d*n,cudaMemcpyHostToDevice);
      QR(d_B,d,n,d_R[i],handle,cusolverH);
      cudaDeviceSynchronize();

      //  这里回到 B
      cudaMemcpy(B[i],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
    }

    float *d_R12,*d_R34,*d_R1T,*d_R2T,*d_R12T,*d_R34T;
    cudaMalloc((void**)&d_R12,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R34,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R1T,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R2T,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R12T,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R34T,sizeof(float)*2*n*n);

    // R1 R2 转置
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[0],n,&beta,d_R[0],n,d_R1T,n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[1],n,&beta,d_R[1],n,d_R2T,n);

    cublasScopy(handle,n*n,d_R1T,1,d_R12,1);
    cublasScopy(handle,n*n,d_R2T,1,d_R12+n*n,1);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,2*n,n,&alpha,d_R12,n,&beta,d_R12,n,d_R12T,2*n);

    // R3 R4
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[2],n,&beta,d_R[2],n,d_R1T,n);//复用一次
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[3],n,&beta,d_R[3],n,d_R2T,n);

    cublasScopy(handle,n*n,d_R1T,1,d_R34,1);
    cublasScopy(handle,n*n,d_R2T,1,d_R34+n*n,1);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,2*n,n,&alpha,d_R34,n,&beta,d_R34,n,d_R34T,2*n);
    cudaDeviceSynchronize();
       
    QR(d_R12T,2*n,n,d_R[0],handle,cusolverH);
    QR(d_R34T,2*n,n,d_R[1],handle,cusolverH); //结果复用在 d_R[0] 和d_R[1]里面，下面两个R再次合并


    float *d_R1234,*d_R1234T;
    cudaMalloc((void**)&d_R1234,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R1234T,sizeof(float)*2*n*n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[0],n,&beta,d_R[0],n,d_R1T,n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[1],n,&beta,d_R[1],n,d_R2T,n);
    cublasScopy(handle,n*n,d_R1T,1,d_R1234,1);
    cublasScopy(handle,n*n,d_R2T,1,d_R1234+n*n,1);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,2*n,n,&alpha,d_R1234,n,&beta,d_R1234,n,d_R1234T,2*n);

    QR(d_R1234T,2*n,n,d_R[0],handle,cusolverH); //最后的要分解的R 存放在R【0】中
   // cout<<"QR decom zhi qian de  R:"<<endl; printTensor(d_R1234T,4,4,1);

    //1、将d_R12T分为垂直方向分为两个n*n R1 R2
    float *d_R1,*d_R2,*d_R3,*d_R4;
    cudaMalloc((void**)&d_R1,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R2,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R3,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R4,sizeof(float)*n*n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,2*n,&alpha,d_R12T,2*n,&beta,d_R12T,2*n,d_R12,n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,2*n,&alpha,d_R34T,2*n,&beta,d_R34T,2*n,d_R34,n);
    cublasScopy(handle,n*n,d_R12,1,d_R1,1);
    cublasScopy(handle,n*n,d_R12+n*n,1,d_R2,1);
    cublasScopy(handle,n*n,d_R34,1,d_R3,1);
    cublasScopy(handle,n*n,d_R34+n*n,1,d_R4,1);

    //2、d_R1234T同样垂直方向分为两个矩阵 
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,2*n,&alpha,d_R1234T,2*n,&beta,d_R1234T,2*n,d_R1234,n);
    cublasScopy(handle,n*n,d_R1234,1,d_R1T,1);
    cublasScopy(handle,n*n,d_R1234+n*n,1,d_R2T,1);

    //d_B[i] * d_Ri.T  因此d_Ri 是通过转置之后得到的，因此乘法时要转置,结果仍存放在d_B[i]中

    cudaMemcpy(d_B,B[0],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R1,n,&beta,d_B,d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R1T,n,&beta,d_B,d);
    cudaMemcpy(B[0],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);

    cudaMemcpy(d_B,B[1],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R2,n,&beta,d_B,d);
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R1T,n,&beta,d_B,d);
    cudaMemcpy(B[1],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);

    cudaMemcpy(d_B,B[2],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R3,n,&beta,d_B,d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R2T,n,&beta,d_B,d);
    cudaMemcpy(B[2],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);

    cudaMemcpy(d_B,B[3],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R4,n,&beta,d_B,d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R2T,n,&beta,d_B,d);
    cudaMemcpy(B[3],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);
    /*cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[0],d,d_R1,n,&beta,d_B[0],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[1],d,d_R2,n,&beta,d_B[1],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[2],d,d_R3,n,&beta,d_B[2],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[3],d,d_R4,n,&beta,d_B[3],d);*/
    /*cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[0],d,d_R1T,n,&beta,d_B[0],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[1],d,d_R1T,n,&beta,d_B[1],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[2],d,d_R2T,n,&beta,d_B[2],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[3],d,d_R2T,n,&beta,d_B[3],d);*/
    //2、对d_R[0],即求得最后的R(n*n) svd 分解,结果保存在d_Rk中
    float *d_Rk,*d_out[4];
    cudaMalloc((void**)&d_Rk,sizeof(float)*n*k);
    svd(d_R[0],d_Rk,n,n,k,handle,cusolverH); //d_R[0]  都是0

    for(unsigned i = 0; i < p; ++i) {
      /* code */
      cudaMalloc((void**)&d_out[i],sizeof(float)*d*k);
      cudaMemcpy(d_B,B[i],sizeof(float)*d*n,cudaMemcpyHostToDevice);
      cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,k,n,&alpha,d_B,d,d_Rk,n,&beta,d_out[i],d);
      //cudaFree(d_B[i]);
      cudaFreeHost(B[i]);
      cudaFree(d_R[i]);
    }

    float *d_tmp,*d_UT;
    cudaMalloc((void**)&d_tmp,sizeof(float)*d*k);
    cudaMalloc((void**)&d_UT,sizeof(float)*m*k);
   
    for(unsigned i = 0; i < p; ++i) {
      cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,k,d,&alpha,d_out[i],d,&beta,d_out[i],d,d_tmp,k);
      cublasScopy(handle,k*d,d_tmp,1,d_UT+i*d*k,1);
    }
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,k,&alpha,d_UT,k,&beta,d_UT,k,d_U,m);

    cudaFree(d_R12);
    cudaFree(d_R34);
    cudaFree(d_R1T);
    cudaFree(d_R2T);
    cudaFree(d_R1234);
    cudaFree(d_R1234T);
    cudaFree(d_tmp);
    cudaFree(d_UT);
    cudaFree(d_R1);
    cudaFree(d_R2);
    cudaFree(d_R3);
    cudaFree(d_R4);
}

void tsqr_svd_half(float *A,int a,int b,int c,int k,float *d_U,cublasHandle_t handle, cusolverDnHandle_t cusolverH)
{
  

    long m = a*b;
    int n=c;

    int p=4;//将A按照垂直方向分为p块，每块size (m/4)*n
    int d = m/4;

    float *B[4];
    cout<<"start tsqr_svd_half function"<<endl;

    cudaHostAlloc((void**)&B[0],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[1],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[2],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[3],sizeof(float)*d*n,0);

    long long row,col;
    for(long long i = 0; i < m*n; ++i) {
      row = i%m;
      col = i/m;
      if(row/d <1)
      {
        B[0][row+col*d] = A[i];
      }else if(row/d<2 && row/d>=1)
      {
        B[1][row-d+col*d] = A[i];
      }else if(row/d<3 && row/d>=2)
      {
        B[2][row-2*d+col*d] = A[i];
      }else if(row/d<4 && row/d>=3)
      {
        B[3][row-3*d+col*d] = A[i];
      }     
    }

    cout<<"aaaaaaaa"<<endl;

    float alpha = 1.0;
    float beta = 0.0;
    float *d_B;
    cudaMalloc((void**)&d_B,sizeof(float)*d*n);

    float *d_R[4];
   // printTensor(d_B[0],d,n,1);
    for(int i =0;i<p;i++)
    {
      cudaMalloc((void**)&d_R[i],sizeof(float)*n*n); 
      cudaMemcpy(d_B,B[i],sizeof(float)*d*n,cudaMemcpyHostToDevice);
      QR(d_B,d,n,d_R[i],handle,cusolverH);
      cudaDeviceSynchronize();
      //  这里回到 B
      cudaMemcpy(B[i],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
    }

    float *d_R12,*d_R34,*d_R1T,*d_R2T,*d_R12T,*d_R34T;
    cudaMalloc((void**)&d_R12,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R34,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R1T,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R2T,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R12T,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R34T,sizeof(float)*2*n*n);

    // R1 R2 转置
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[0],n,&beta,d_R[0],n,d_R1T,n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[1],n,&beta,d_R[1],n,d_R2T,n);

    cublasScopy(handle,n*n,d_R1T,1,d_R12,1);
    cublasScopy(handle,n*n,d_R2T,1,d_R12+n*n,1);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,2*n,n,&alpha,d_R12,n,&beta,d_R12,n,d_R12T,2*n);

    // R3 R4
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[2],n,&beta,d_R[2],n,d_R1T,n);//复用一次
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[3],n,&beta,d_R[3],n,d_R2T,n);

    cublasScopy(handle,n*n,d_R1T,1,d_R34,1);
    cublasScopy(handle,n*n,d_R2T,1,d_R34+n*n,1);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,2*n,n,&alpha,d_R34,n,&beta,d_R34,n,d_R34T,2*n);
    cudaDeviceSynchronize();
       
    QR(d_R12T,2*n,n,d_R[0],handle,cusolverH);
    QR(d_R34T,2*n,n,d_R[1],handle,cusolverH); //结果复用在 d_R[0] 和d_R[1]里面，下面两个R再次合并

    float *d_R1234,*d_R1234T;
    cudaMalloc((void**)&d_R1234,sizeof(float)*2*n*n);
    cudaMalloc((void**)&d_R1234T,sizeof(float)*2*n*n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[0],n,&beta,d_R[0],n,d_R1T,n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,n,&alpha,d_R[1],n,&beta,d_R[1],n,d_R2T,n);
    cublasScopy(handle,n*n,d_R1T,1,d_R1234,1);
    cublasScopy(handle,n*n,d_R2T,1,d_R1234+n*n,1);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,2*n,n,&alpha,d_R1234,n,&beta,d_R1234,n,d_R1234T,2*n);

    QR(d_R1234T,2*n,n,d_R[0],handle,cusolverH); //最后的要分解的R 存放在R【0】中

    //1、将d_R12T分为垂直方向分为两个n*n R1 R2
    float *d_R1,*d_R2,*d_R3,*d_R4;
    cudaMalloc((void**)&d_R1,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R2,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R3,sizeof(float)*n*n);
    cudaMalloc((void**)&d_R4,sizeof(float)*n*n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,2*n,&alpha,d_R12T,2*n,&beta,d_R12T,2*n,d_R12,n);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,2*n,&alpha,d_R34T,2*n,&beta,d_R34T,2*n,d_R34,n);
    /*
    cublasScopy(handle,n*n,d_R12,1,d_R1,1);
    cublasScopy(handle,n*n,d_R12+n*n,1,d_R2,1);
    cublasScopy(handle,n*n,d_R34,1,d_R3,1);
    cublasScopy(handle,n*n,d_R34+n*n,1,d_R4,1);*/

    //2、d_R1234T同样垂直方向分为两个矩阵 
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,2*n,&alpha,d_R1234T,2*n,&beta,d_R1234T,2*n,d_R1234,n);
   /* cublasScopy(handle,n*n,d_R1234,1,d_R1T,1);
    cublasScopy(handle,n*n,d_R1234+n*n,1,d_R2T,1);*/

    //d_B[i] * d_Ri.T  因此d_Ri 是通过转置之后得到的，因此乘法时要转置,结果仍存放在d_B[i]中
    half *h_R[3],*h_B;
    cudaMalloc((void**)&h_B,sizeof(half)*d*n);
    for(unsigned i = 0; i < 3; ++i) {
        cudaMalloc((void**)&h_R[i],sizeof(half)*2*n*n);
    }
   /* f2h(d_R1,h_R[0],n*n);
    f2h(d_R2,h_R[1],n*n);
    f2h(d_R3,h_R[2],n*n);
    f2h(d_R4,h_R[3],n*n);
    f2h(d_R1T,h_R[4],n*n);
    f2h(d_R2T,h_R[5],n*n);*/
    f2h(d_R12,h_R[0],2*n*n);
    f2h(d_R34,h_R[1],2*n*n);
    f2h(d_R1234,h_R[2],2*n*n);

//-------------------
    cudaMemcpy(d_B,B[0],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R1,n,&beta,d_B,d);
    /*cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[0],CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);*/
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[0],CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R1T,n,&beta,d_B,d);
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[2],CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy(B[0],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);
   //cout<<"zui hou de Q0:"<<endl;printTensor(d_B,3,3,1);
//--------------
    cudaMemcpy(d_B,B[1],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R2,n,&beta,d_B,d);
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[0]+n*n,CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R1T,n,&beta,d_B,d);
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[2],CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy(B[1],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);
    //cout<<"zui hou de Q1:"<<endl;printTensor(d_B,3,3,1);
//------------------------
    cudaMemcpy(d_B,B[2],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R3,n,&beta,d_B,d);
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[1],CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R2T,n,&beta,d_B,d);
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[2]+n*n,CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy(B[2],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);
    //cout<<"zui hou de Q2:"<<endl;printTensor(d_B,3,3,1);
//---------------------
    cudaMemcpy(d_B,B[3],sizeof(float)*d*n,cudaMemcpyHostToDevice);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R4,n,&beta,d_B,d);
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[1]+n*n,CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    f2h(d_B,h_B,d*n);
    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B,d,d_R2T,n,&beta,d_B,d);
     cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           d,n,n,
                           &alpha,h_B,CUDA_R_16F,d,
                           h_R[2]+n*n,CUDA_R_16F,n,
                           &beta,d_B,CUDA_R_32F,d,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
     //cout<<"zui hou de Q:"<<endl;printTensor(d_B,3,3,1);
    cudaMemcpy(B[3],d_B,sizeof(float)*d*n,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//--------------------------------
    //2、对d_R[0],即求得最后的R(n*n) svd 分解,结果保存在d_Rk中
    float *d_Rk,*d_out[4];
    cudaMalloc((void**)&d_Rk,sizeof(float)*n*k);
    svd(d_R[0],d_Rk,n,n,k,handle,cusolverH);

    for(unsigned i = 0; i < p; ++i) {
      cudaMalloc((void**)&d_out[i],sizeof(float)*d*k);
      cudaMemcpy(d_B,B[i],sizeof(float)*d*n,cudaMemcpyHostToDevice);
      cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,k,n,&alpha,d_B,d,d_Rk,n,&beta,d_out[i],d);
       
      //cudaFree(d_B[i]);
      cudaFreeHost(B[i]);
      cudaFree(d_R[i]);
    }
   // cout<<"d_out de value:"<<endl;printTensor(d_out[0],3,3,1);

    float *d_tmp,*d_UT;
    cudaMalloc((void**)&d_tmp,sizeof(float)*d*k);
    cudaMalloc((void**)&d_UT,sizeof(float)*m*k);
   
    for(unsigned i = 0; i < p; ++i) {
      cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,k,d,&alpha,d_out[i],d,&beta,d_out[i],d,d_tmp,k);
      cublasScopy(handle,k*d,d_tmp,1,d_UT+i*d*k,1);
    }
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,k,&alpha,d_UT,k,&beta,d_UT,k,d_U,m);

    for(unsigned i = 0; i < 3; ++i) {
      cudaFree(h_R[i]);
    }
    cudaFree(h_B);
    cudaFree(d_R12);
    cudaFree(d_R34);
    cudaFree(d_R1T);
    cudaFree(d_R2T);
    cudaFree(d_R1234);
    cudaFree(d_R1234T);
    cudaFree(d_tmp);
    cudaFree(d_UT);
    cudaFree(d_R1);
    cudaFree(d_R2);
    cudaFree(d_R3);
    cudaFree(d_R4);
}
