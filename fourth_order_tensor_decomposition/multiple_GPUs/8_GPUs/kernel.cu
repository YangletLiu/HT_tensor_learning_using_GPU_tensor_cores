#include "head.h"

void printTensor(float *d_des,long m,long n,long l){
  float *des = new float[m*n*l]();
  cudaMemcpy(des,d_des,sizeof(float)*m*n*l,cudaMemcpyDeviceToHost);
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

__global__  void floattohalf(float *AA,half *BB,long m){
  long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long temp = blockDim.x*gridDim.x;
  if(i<m){
    BB[i]=__float2half(AA[i]);
    i+=temp;
  }
  __syncthreads();
}

void f2h(float *A,half *B,long num){
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
__global__ void mode2(float *A,float *B,long m,long n,long r)
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
__global__ void transmission(float *d_A,float *d_B,long a,long b)
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


void genHtensor(float *X,long a,long b,long c,long d)
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
   float *U5,*U4,*U3,*U6,*B3,*B2,*B1;
   cudaHostAlloc((void**)&U6,sizeof(float)*size*k[6],0);
   cudaHostAlloc((void**)&U5,sizeof(float)*size*k[5],0);
   cudaHostAlloc((void**)&U4,sizeof(float)*size*k[4],0);
   cudaHostAlloc((void**)&U3,sizeof(float)*size*k[3],0);
   cudaHostAlloc((void**)&B3,sizeof(float)*k[5]*k[6]*k[2],0);
   cudaHostAlloc((void**)&B2,sizeof(float)*k[3]*k[4]*k[1],0);
   cudaHostAlloc((void**)&B1,sizeof(float)*k[1]*k[2]*k[0],0);

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
   float *d_U6,*d_U5,*d_U4,*d_U3,*d_B3,*d_B2,*d_B1;
   cudaMalloc((void**)&d_U6,sizeof(float)*size*k[6]);
   cudaMalloc((void**)&d_U5,sizeof(float)*size*k[5]);
   cudaMalloc((void**)&d_U4,sizeof(float)*size*k[4]);
   cudaMalloc((void**)&d_U3,sizeof(float)*size*k[3]);
   cudaMalloc((void**)&d_B3,sizeof(float)*k[5]*k[6]*k[2]);
   cudaMalloc((void**)&d_B2,sizeof(float)*k[3]*k[4]*k[1]);
   cudaMalloc((void**)&d_B1,sizeof(float)*k[1]*k[2]*k[0]);

   cudaMemcpy(d_U6,U6,sizeof(float)*size*k[6],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U5,U5,sizeof(float)*size*k[5],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U4,U4,sizeof(float)*size*k[4],cudaMemcpyHostToDevice);
   cudaMemcpy(d_U3,U3,sizeof(float)*size*k[3],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B3,B3,sizeof(float)*k[5]*k[6]*k[2],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B2,B2,sizeof(float)*k[3]*k[4]*k[1],cudaMemcpyHostToDevice);
   cudaMemcpy(d_B1,B1,sizeof(float)*k[1]*k[2]*k[0],cudaMemcpyHostToDevice);

   cublasHandle_t handle;
   cublasCreate(&handle);
   float alpha = 1.0;
   float beta = 0.0;

   float *d_U2,*d_X,*d_U1;
   cudaMalloc((void**)&d_U2,sizeof(float)*size*size*k[2]);
   cudaMalloc((void**)&d_U1,sizeof(float)*size*size*k[1]);
   cudaMalloc((void**)&d_X,sizeof(float)*size*size*size*size);

   float*d_U5B3;
   cudaMalloc((void**)&d_U5B3, sizeof(float)*size*k[6]*k[2]);
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
   float *d_U3B2;
   cudaMalloc((void**)&d_U3B2, sizeof(float)*size*k[4]*k[1]);
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
 float *d_U1B1;
 cudaMalloc((void**)&d_U1B1,sizeof(float)*size*size*k[2]);
   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,size*size,k[2],k[1],
               &alpha,d_U1,size*size,d_B1,k[1],
               &beta,d_U1B1,size*size
               );
  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,size*size,size*size,k[2],
              &alpha,d_U1B1,size*size,d_U2,size*size,
              &beta,d_X,size*size
              );
  cudaDeviceSynchronize();
  cudaMemcpy(X,d_X,sizeof(float)*size*size*size*size,cudaMemcpyDeviceToHost);
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

void gpu0_u1(float* d_X,int a,int b,int c,int d,int k,int gpu0,float* d_Ux1,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{

  float alpha = 1.0;
  float beta = 0.0;


  dim3 block0((a*b*c*d+1024-1)/1024,1,1);
  dim3 threads(1024,1,1);

  float *d_X1_X1;
  cudaMalloc((void**)&d_X1_X1,sizeof(float)*a*a);
  cudaDeviceSynchronize();

  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*c*d,
                &alpha,d_X,a,d_X,a,
                &beta,d_X1_X1,a
                );

  eig(d_X1_X1,a,a,cusolverH); 
  cublasScopy(handle,a*k,d_X1_X1+a*(a-k),1,d_Ux1,1);
  cudaFree(d_X1_X1); 
}

void gpu1_u2(float* d_X,int a,int b,int c,int d,int k,int gpu1,float* d_Ux2,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{

  float alpha = 1.0;
  float beta = 0.0;


  dim3 block0((a*b*c*d+1024-1)/1024,1,1);
  dim3 threads(1024,1,1);

  float *d_X2_X2,*d_X2;
  cudaMalloc((void**)&d_X2_X2,sizeof(float)*b*b);
  cudaDeviceSynchronize();

  cudaMalloc((void**)&d_X2,sizeof(float)*a*b*c*d);
  mode2<<<block0,threads>>>(d_X,d_X2,a,b,c*d);

 cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a*c*d,
                &alpha,d_X2,b,d_X2,b,
                &beta,d_X2_X2,b
                );

  eig(d_X2_X2,b,b,cusolverH); 
  cublasScopy(handle,b*k,d_X2_X2+b*(b-k),1,d_Ux2,1);
  cudaFree(d_X2_X2);
  cudaFree(d_X2);
}

void gpu2_u3(float* d_X,int a,int b,int c,int d,int k,int gpu0,float* d_Ux3,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{

  float alpha = 1.0;
  float beta = 0.0;
 

  dim3 block0((a*b*c*d+1024-1)/1024,1,1);
  dim3 threads(1024,1,1);

  float *d_X3_X3,*d_X3;
  cudaMalloc((void**)&d_X3_X3,sizeof(float)*c*c);
  cudaMalloc((void**)&d_X3,sizeof(float)*a*b*c*d);

  mode2<<<block0,threads>>>(d_X,d_X3,a*b,c,d);

  cudaDeviceSynchronize();

  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,c,c,a*b*d,
                &alpha,d_X3,c,d_X3,c,
                &beta,d_X3_X3,c
                );

  eig(d_X3_X3,c,c,cusolverH);  
  cublasScopy(handle,c*k,d_X3_X3+c*(c-k),1,d_Ux3,1);
  cudaFree(d_X3_X3);
  cudaFree(d_X3);

}

void gpu3_u4(float* d_X,int a,int b,int c,int d,int k,int gpu0,float* d_Ux4,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{

  float alpha = 1.0;
  float beta = 0.0;
 
  dim3 block0((a*b*c*d+1024-1)/1024,1,1);
  dim3 threads(1024,1,1);

  float *d_X4_X4;

  cudaMalloc((void**)&d_X4_X4,sizeof(float)*d*d);
  cudaDeviceSynchronize();

  cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,d,d,a*c*b,
                &alpha,d_X,a*b*c,d_X,a*b*c,
                &beta,d_X4_X4,d
                );

  eig(d_X4_X4,d,d,cusolverH);  
  cublasScopy(handle,d*k,d_X4_X4+d*(d-k),1,d_Ux4,1);
  cudaFree(d_X4_X4);
}

void gpu4_u5(float* d_X,int a,int b,int c,int d,int k,int gpu4,float* d_Ux5,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{


  dim3 block_t((a*b*k+1024-1)/1024,1,1);
  dim3 threads(1024,1,1);
  float *d_U;
  cudaMalloc((void**)&d_U,sizeof(float)*a*b*k);

  rsvd(d_X,d_U,a*b,c*d,k,handle,cusolverH);
  transmission<<<block_t,threads>>>(d_U,d_Ux5,a*b,k);
  cudaFree(d_U);
}

void gpu5_u6(float* d_X,int a,int b,int c,int d,int k,int gpu5,float* d_Ux6,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{

   float alpha = 1.0;
  float beta = 0.0;

  float* d_XT;
  cudaMalloc((void**)&d_XT,sizeof(float)*a*b*c*d);
  cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,c*d,a*b,&alpha,d_X,a*b,&beta,d_X,a*b,d_XT,c*d);
  dim3 block_t((a*b*k+1024-1)/1024,1,1);
  dim3 threads(1024,1,1);
  float *d_U;
  cudaMalloc((void**)&d_U,sizeof(float)*a*b*k);

  rsvd(d_XT,d_U,c*d,a*b,k,handle,cusolverH);
  transmission<<<block_t,threads>>>(d_U,d_Ux6,c*d,k);
  cudaFree(d_U);
  cudaFree(d_XT);
}


void recover(float *d_Ux1,float *d_Ux2,float *d_Ux3,float *d_Ux4,
             float *d_B1,float *d_B2,float *d_B3,float *d_r,
             int a,int b,int c,int d,int k,cublasHandle_t handle)
{

   float alpha = 1.0;
  float beta = 0.0;
 

  float *d_U4B2,*d_U2_r;
  cudaMalloc((void**)&d_U4B2,sizeof(float)*a*k*k);
  cudaMalloc((void**)&d_U2_r,sizeof(float)*a*b*k);

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
              a,k*k,k,
              &alpha,d_Ux1,a,d_B2,k,
              &beta,d_U4B2,a
                );
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                              a,b,k,
                              &alpha,d_U4B2,a,a*k,d_Ux2,b,0,
                              &beta,d_U2_r,a,a*b,k
                              );
    float *d_U6B3,*d_U3_r;
    cudaMalloc((void**)&d_U6B3,sizeof(float)*c*k*k);
    cudaMalloc((void**)&d_U3_r,sizeof(float)*c*d*k);

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
              c,k*k,k,
              &alpha,d_Ux3,c,d_B3,k,
              &beta,d_U6B3,c
                );

    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                              c,d,k,
                              &alpha,d_U6B3,c,c*k,d_Ux4,d,0,
                              &beta,d_U3_r,c,c*d,k
                              );
    float *d_U2B1,*d_X_r;
    cudaMalloc((void**)&d_U2B1,sizeof(float)*a*b*k);
    cudaMalloc((void**)&d_X_r,sizeof(float)*a*b*c*d);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
              a*b,k,k,
              &alpha,d_U2_r,a*b,d_B1,k,&beta,d_U2B1,a*b
              );    
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                 a*b,c*d,k,
                 &alpha,d_U2B1,a*b,d_U3_r,c*d,
                 &beta,d_r,a*b
                 );

}


void ttm(float *d_U1,float *d_U2,float *d_U3,float *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle)
{
//d_Ux3,d_Ux6,d_Ux7,d_B3,c,d,k[2],k[5],k[6]
// U1 a*b*k1 , U2  a*k2 , U3  b*k3
  float alpha = 1.0;
  float beta = 0.0;

  float *d_U1U2;
  cudaMalloc((void**)&d_U1U2,sizeof(float)*k2*b*k1);

 //cout<<"ttm function canshu 1:"<<endl;printTensor(d_U1,4,4,1);
 // cout<<"ttm function canshu 2:"<<endl;printTensor(d_U2,4,4,1);
 //  cout<<"ttm function canshu 3:"<<endl;printTensor(d_U3,4,4,1);

  cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
              k2,b*k1,a,
              &alpha,d_U2,a,d_U1,a,
              &beta,d_U1U2,k2
                );

  cublasSgemmStridedBatched(handle,
                            CUBLAS_OP_N,CUBLAS_OP_N,
                            k2,k3,b,
                            &alpha,d_U1U2,k2,k2*b,d_U3,b,0,
                            &beta,d_B,k2,k2*k3,k1                           
                            );
  cudaDeviceSynchronize();
  //cout<<"ttm function result:"<<endl;printTensor(d_B,4,4,1);
  cudaFree(d_U1U2);
}
void ttm_tensorcore(half *d_U1,half *d_U2,half *d_U3,float *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle)
{

  float alpha = 1.0;
  float beta = 0.0;

  float *d_U1U2;
  cudaMalloc((void**)&d_U1U2,sizeof(float)*k2*b*k1);
  half *d_U1U2_h;
  cudaMalloc((void**)&d_U1U2_h,sizeof(half)*k2*b*k1);

  cublasGemmEx(handle,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
              k2,b*k1,a,
                            &alpha,d_U2,CUDA_R_16F,a,
                           d_U1,CUDA_R_16F,a,
                           &beta,d_U1U2,CUDA_R_32F,k2,
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  /*cublasSgemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           k2,b*k1,a,
                           &alpha,d_U2,CUDA_R_16F,a,
                           d_U1,CUDA_R_16F,a,
                           &beta,d_U1U2,CUDA_R_32F,k2);*/
  //cout<<"TTM zhong de di yi bu :-----------"<<endl;printTensor(d_U1U2,4,4,1);
  f2h(d_U1U2,d_U1U2_h,k2*b*k1);
  cudaDeviceSynchronize();


  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_N,
                 k2,k3,b,
                 &alpha,d_U1U2_h,CUDA_R_16F,k2,k2*b,
                 d_U3,CUDA_R_16F,b,0,
                 &beta,d_B,CUDA_R_32F,k2,k2*k3,
                 k1,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
