#include "head.h"

void printTensor(dt *d_des,long m,long n,long l,long k){
  dt *des = new dt[m*n*l*k]();
  cudaMemcpy(des,d_des,sizeof(dt)*m*n*l*k,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for(int d=0;d<k;d++){
  for(int c = 0;c<l;c++){
    for(int b = 0;b<n;b++){
      for(int a = 0;a<m;a++){
        cout<<des[d*m*n*l+c*m*n+b*m+a]<<" ";
      }
      cout<<endl;
    }
    cout<<"~~~~~~~~~~~~~~~~~~~~~"<<endl;
  }
}
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

__global__ void sub(dt *A,dt *B,long a,long b,long c)
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;
	while(i<a*b*c)
	{
		B[i] = A[i] - B[i];
		i+=temp;
	}
	__syncthreads();
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

void genHtensor(dt *X,long a,long b,long c,long d)
{	
	 srand((unsigned)time(NULL)); 
   int size=a;
   int k[7];
   int q=7;
   int w=3;
   for(int i =0;i<7;i++){
        k[i]=(rand() % (q-w+1))+ w; //3-10随机整数
       //k[i]=size;             
   }
   k[0]=1;
   dt *U5,*U4,*U3,*U6,*B3,*B2,*B1;
   cudaHostAlloc((void**)&U6,sizeof(dt)*size*k[6],0);
   cudaHostAlloc((void**)&U5,sizeof(dt)*size*k[5],0);
   cudaHostAlloc((void**)&U4,sizeof(dt)*size*k[4],0);
   cudaHostAlloc((void**)&U3,sizeof(dt)*size*k[3],0);
   cudaHostAlloc((void**)&B3,sizeof(dt)*k[5]*k[6]*k[2],0);
   cudaHostAlloc((void**)&B2,sizeof(dt)*k[3]*k[4]*k[1],0);
   cudaHostAlloc((void**)&B1,sizeof(dt)*k[1]*k[2]*k[0],0);

   for(long i=0;i<size*k[6];i++)
   {
        U6[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<size*k[5];i++)
   {
        U5[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<size*k[4];i++)
   {
        U4[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<size*k[3];i++)
   {
        U3[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<k[5]*k[6]*k[2];i++)
   {
        B3[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<k[3]*k[4]*k[1];i++)
   {
        B2[i]=rand()*2.0/RAND_MAX - 1.0;
   }
   for(long i=0;i<k[1]*k[2]*k[0];i++)
   {
        B1[i]=rand()*2.0/RAND_MAX - 1.0;
   }


   dt *d_U6,*d_U5,*d_U4,*d_U3,*d_B3,*d_B2,*d_B1;
   cudaMalloc((void**)&d_U6,sizeof(dt)*size*k[6]);
   cudaMalloc((void**)&d_U5,sizeof(dt)*size*k[5]);
   cudaMalloc((void**)&d_U4,sizeof(dt)*size*k[4]);
   cudaMalloc((void**)&d_U3,sizeof(dt)*size*k[3]);
   cudaMalloc((void**)&d_B3,sizeof(dt)*k[5]*k[6]*k[2]);
   cudaMalloc((void**)&d_B2,sizeof(dt)*k[3]*k[4]*k[1]);
   cudaMalloc((void**)&d_B1,sizeof(dt)*k[1]*k[2]*k[0]);

   cudaMemcpy(d_U6,U6,sizeof(dt)*size*k[6],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U5,U5,sizeof(dt)*size*k[5],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U4,U4,sizeof(dt)*size*k[4],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U3,U3,sizeof(dt)*size*k[3],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B3,B3,sizeof(dt)*k[5]*k[6]*k[2],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B2,B2,sizeof(dt)*k[3]*k[4]*k[1],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B1,B1,sizeof(dt)*k[1]*k[2]*k[0],cudaMemcpyHostToDevice);

   cublasHandle_t handle;
   cublasCreate(&handle);
   dt alpha = 1.0;
   dt beta = 0.0;

   dt *d_U2,*d_X,*d_U1;
   cudaMalloc((void**)&d_U2,sizeof(dt)*size*size*k[2]);
   cudaMalloc((void**)&d_U1,sizeof(dt)*size*size*k[1]);
   cudaMalloc((void**)&d_X,sizeof(dt)*size*size*size*size);

   dt*d_U5B3;
   cudaMalloc((void**)&d_U5B3, sizeof(dt)*size*k[6]*k[2]);
   //ttm B2 x1 U5 x2 U6
   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
               size,k[6]*k[2],k[5],
               &alpha,d_U5,size,d_B3,k[5],
               &beta,d_U5B3,size
               );
   cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                             size,size,k[6],
                             &alpha,d_U5B3,size,size*k[6],d_U6,size,0,
                             &beta,d_U2,size,size,k[2]
                             );
   //ttm B1 x1 U3 x2 U4
   dt *d_U3B2;
   cudaMalloc((void**)&d_U3B2, sizeof(dt)*size*k[4]*k[1]);
   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
               size,k[4]*k[1],k[3],
               &alpha,d_U3,size,d_B2,k[3],
               &beta,d_U3B2,size
               );
   cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
               size,size,k[4],
               &alpha,d_U3B2,size,size*k[4],d_U4,size,0,
               &beta,d_U1,size,size*size,k[1]
               );
   cudaDeviceSynchronize();

   // ttm B1 x1 U1 x2 U2
 dt *d_U1B1;
 cudaMalloc((void**)&d_U1B1,sizeof(dt)*size*size*k[2]);
   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,size*size,k[2],k[1],
               &alpha,d_U1,size*size,d_B1,k[1],
               &beta,d_U1B1,size*size
               );
  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,size*size,size*size,k[2],
              &alpha,d_U1B1,size*size,d_U2,size*size,
              &beta,d_X,size*size
              );
  cudaDeviceSynchronize();
  
  //printTensor(d_X,2,2,2,2);

  cudaMemcpy(X,d_X,sizeof(dt)*size*size*size*size,cudaMemcpyDeviceToHost);
   cudaFreeHost(U6);
   cudaFreeHost(U5);
   cudaFreeHost(U4);
   cudaFreeHost(U3);
   cudaFreeHost(B3);
   cudaFreeHost(B2);
   cudaFreeHost(B1);
  
   cudaFree(d_U6);
   cudaFree(d_U5);
   cudaFree(d_U4);
   cudaFree(d_U3);
   cudaFree(d_B3);
   cudaFree(d_B1);
   cudaFree(d_B2);
   cudaFree(d_U5B3);
   cudaFree(d_U3B2);
   cudaFree(d_U1B1);
   cudaFree(d_X);
   cublasDestroy(handle);
}
__global__ void upper(dt *A,dt *R,int m,int n)
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
     //printf("after geqrf: info_gpu = %d\n", info_gpu);

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
/*void rsvd(float *d_A,float *d_U,int m,int n,int ks,cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{
    int p=10;
    float alpha = 1.0;
    float beta =0.0;
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    float *d_B,*d_C;
    cudaMalloc((void**)&d_B, sizeof(float)*n*ks);
    cudaMalloc((void**)&d_C,sizeof(float)*m*ks);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_B, n*ks, 0, 1);
    //printTensor(d_B,3,3,1,1);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, m, ks, n,&alpha,d_A,m,d_B,n,&beta,d_C,m);

    QR(d_C,m,ks,cusolverH);
    for(int i=0;i<p;i++)
    {
        cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n, ks, m,&alpha,d_A,m,d_C,m,&beta,d_B,n);
        QR(d_B,n,ks,cusolverH);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, ks, n,&alpha,d_A, m, d_B, n,&beta,d_C,m);
        QR(d_C,m,ks,cusolverH);
    }
    cublasSgemm(handle,CUBLAS_OP_T,  CUBLAS_OP_N,ks, n, m,&alpha,d_C, m, d_A,  m,&beta,d_B, ks);
     float *d_UT,*d_S,*d_V;
    cudaMalloc((void**)&d_UT, sizeof(float)*ks*ks);
    cudaMalloc((void**)&d_S,sizeof(float)*ks);
    cudaMalloc((void**)&d_V,sizeof(float)*n*n);

    svd(d_B,ks,n,d_UT, d_S, d_V,handle,cusolverH);
    cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,m, ks, ks,&alpha,d_C, m,d_UT, ks,&beta,d_U, m);

    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_UT);
    cudaFree(d_S);
    cudaFree(d_V);
}*/
void rsvd(float *d_A,float *d_U,int m,int n,int ks,cublasHandle_t handle,cusolverDnHandle_t cusolverH)
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