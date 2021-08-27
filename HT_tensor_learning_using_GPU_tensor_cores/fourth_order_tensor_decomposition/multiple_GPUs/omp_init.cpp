#include "head.h"

int main()
{
    // 24 - 168
    int n=168;
    int a=n;
    int b=n;
    int c=n;
    int d=n;
    int k=n*0.5;
    struct timeval start;
    struct timeval end;
    float time_use=0;
    int gpu0 = 0;
    int gpu1 = 1;

    cudaSetDevice(gpu0);
    float *X;
    cudaHostAlloc((void**)&X,sizeof(float)*a*b*c*d,0);
    genHtensor(X,a,b,c,d); //init tensor


    // float *d_Ux4,*d_Ux5,*d_Ux3,*d_Ux2,*d_Ux1,*d_Ux6,*d_X,*d_X_g1;
    // float *d_B1,*d_B2,*d_B3,*d_Ux6_gpu0;
    // half *h_Ux6_gpu0;

    cudaSetDevice(gpu0);
    
     cublasHandle_t handle;
      cublasCreate(&handle);
     
      cusolverDnHandle_t cusolverH = NULL;
      cusolverDnCreate(&cusolverH);

    float *d_Ux1,*d_Ux2,*d_Ux5,*d_Ux6_gpu0,*d_B2,*d_B1,*d_X;
    cudaMalloc((void**)&d_Ux1,sizeof(float)*a*k);
    cudaMalloc((void**)&d_Ux2,sizeof(float)*b*k);
    cudaMalloc((void**)&d_Ux5,sizeof(float)*a*b*k);
    cudaMalloc((void**)&d_B1,sizeof(float)*k*k*1);
    cudaMalloc((void**)&d_Ux6_gpu0,sizeof(float)*c*d*k);
    cudaMalloc((void**)&d_X,sizeof(float)*a*b*c*d);
    cudaMalloc((void**)&d_B2,sizeof(float)*k*k*k);
    
    
    half *h_Ux1,*h_Ux2,*h_Ux5,*h_Ux6_gpu0;
    cudaMalloc((void**)&h_Ux1,sizeof(half)*a*k);
    cudaMalloc((void**)&h_Ux2,sizeof(half)*b*k);
    cudaMalloc((void**)&h_Ux5,sizeof(half)*a*b*k);
    cudaMalloc((void**)&h_Ux6_gpu0,sizeof(half)*c*d*k);
     half *h_X;
     cudaMalloc((void**)&h_X,sizeof(half)*a*b*c*d);

     cudaSetDevice(gpu1);
    
      cublasHandle_t handle2;
      cublasCreate(&handle2);
      //cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
      cusolverDnHandle_t cusolverH2 = NULL;
      cusolverDnCreate(&cusolverH2);

    float *d_Ux3,*d_Ux4,*d_Ux6,*d_B3,*d_X_g1;
    cudaMalloc((void**)&d_Ux3,sizeof(float)*c*k);
    cudaMalloc((void**)&d_Ux4,sizeof(float)*d*k);
    cudaMalloc((void**)&d_Ux6,sizeof(float)*c*d*k);

    cudaMalloc((void**)&d_X_g1,sizeof(float)*a*b*c*d);
    cudaMalloc((void**)&d_B3,sizeof(float)*k*k*k);
    

    half *h_Ux3,*h_Ux4,*h_Ux6;
    cudaMalloc((void**)&h_Ux3,sizeof(half)*c*k);
    cudaMalloc((void**)&h_Ux4,sizeof(half)*d*k);
    cudaMalloc((void**)&h_Ux6,sizeof(half)*c*d*k);


  cout<<"size :"<<n<<endl;
  gettimeofday(&start,NULL);

   
  #pragma omp parallel num_threads(2)
  {//  two thread:no.1 to control gpu0 ; no.2 to control gpu1    
    int cpuid = omp_get_thread_num();
    if(cpuid == 0)
    {

      cudaSetDevice(gpu0);

      //f2h(d_X,h_X,a*b*c*d);
      cudaMemcpyAsync(d_X,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);
      left(d_X,a,b,c,d,k,gpu0,d_Ux1,d_Ux2,d_Ux5,handle,cusolverH);
      // left_TC(d_X,h_X,a,b,c,d,k,gpu0,d_Ux1,d_Ux2,d_Ux5,handle,cusolverH);
      // f2h(d_Ux5,h_Ux5,a*b*k);
      // f2h(d_Ux1,h_Ux1,a*k);
      // f2h(d_Ux2,h_Ux2,b*k);
      ttm(d_Ux5,d_Ux1,d_Ux2,d_B2,a,b,k,k,k,handle);
      //ttm_tensorcore(h_Ux5,h_Ux1,h_Ux2,d_B2,a,b,k,k,k,handle);

    }
    else if(cpuid == 1)
    {
      
      cudaSetDevice(gpu1);
      cudaMemcpyAsync(d_X_g1,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);
  
      right(d_X_g1,a,b,c,d,k,gpu1,d_Ux3,d_Ux4,d_Ux6,handle2,cusolverH2);
      // right_TC(d_X_g1,a,b,c,d,k,gpu1,d_Ux3,d_Ux4,d_Ux6,handle,cusolverH);
      // f2h(d_Ux6,h_Ux6,c*d*k);
      // f2h(d_Ux3,h_Ux3,c*k);
      // f2h(d_Ux4,h_Ux4,d*k);
      ttm(d_Ux6,d_Ux3,d_Ux4,d_B3,c,d,k,k,k,handle2);
      //ttm_tensorcore(h_Ux6,h_Ux3,h_Ux4,d_B3,a,b,k,k,k,handle);
      
      cudaMemcpyPeerAsync(d_Ux6_gpu0,gpu0,d_Ux6,gpu1, sizeof(float)*c*d*k, 0);

    
    }
    // #pragma omp barrier 
   }

   cudaSetDevice(gpu0);

  //cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);

  ttm(d_X,d_Ux5,d_Ux6_gpu0,d_B1,a*b,c*d,1,k,k,handle);


  //ttm_tensorcore(h_X,h_Ux5,h_Ux6_gpu0,d_B1,a*b,c*d,1,k,k,handle);
  gettimeofday(&end,NULL);
  time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
  time_use=time_use/1000000;
  printf("time_use is %.10f\n",time_use);
  //recover
  cudaSetDevice(0);
  float *d_B3_gp0,*d_Ux3_gpu0,*d_Ux4_gpu0,*d_r;
  cudaMalloc((void**)&d_B3_gp0,sizeof(float)*k*k*k);
  cudaMalloc((void**)&d_Ux3_gpu0,sizeof(float)*c*k);
  cudaMalloc((void**)&d_Ux4_gpu0,sizeof(float)*d*k);
  cudaMalloc((void**)&d_r,sizeof(float)*a*b*c*d);

  cudaSetDevice(1);
  cudaMemcpyPeerAsync(d_B3_gp0,gpu0,d_B3,gpu1, sizeof(float)*k*k*k, 0);
  cudaMemcpyPeerAsync(d_Ux3_gpu0,gpu0,d_Ux3,gpu1, sizeof(float)*c*k, 0);
  cudaMemcpyPeerAsync(d_Ux4_gpu0,gpu0,d_Ux4,gpu1, sizeof(float)*d*k, 0);

  cudaSetDevice(0);
  float alpha1=-1.0;
  float re=0.0;
  float before = 0.0;

  recover(d_Ux1,d_Ux2,d_Ux3_gpu0,d_Ux4_gpu0,d_B1,d_B2,d_B3_gp0,d_r,a,b,c,d,k,handle);
  cublasSaxpy(handle,a*b*c*d,&alpha1,d_X,1,d_r,1); 
  cudaDeviceSynchronize();

  cublasSnrm2(handle,a*b*c*d,d_r,1,&re);
  cublasSnrm2(handle,a*b*c*d,d_X,1,&before);
  cudaDeviceSynchronize();

  cout<<"error rate "<<re/before<<endl;
  cudaDeviceReset();
  return 0;

}



