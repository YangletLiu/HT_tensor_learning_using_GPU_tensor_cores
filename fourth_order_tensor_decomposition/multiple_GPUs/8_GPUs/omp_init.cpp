#include "head.h"

int main()
{
    // 24 - 168
    int n=24;
    int a=n;
    int b=n;
    int c=n;
    int d=n;
    int k=n*0.5;
    struct timeval start;
    struct timeval end;
    float time_use=0;

    float *X;
    cudaHostAlloc((void**)&X,sizeof(float)*a*b*c*d,0);
    genHtensor(X,a,b,c,d); //init tensor


    int gpu0 = 0;
    int gpu1 = 1;
    int gpu2 = 2;
    int gpu3 = 3;
    int gpu4 = 4;
    int gpu5 = 5;
    

    float *d_Ux4,*d_Ux5,*d_Ux3,*d_Ux2,*d_Ux1,*d_Ux6,*d_X,*d_X_g1;
    float *d_B1,*d_B2,*d_B3;
    float *d_Ux2_gpu0,*d_Ux4_gpu2;
    float *d_X_g2,*d_X_g3,*d_X_g4,*d_X_g5,*d_Ux5_gpu0,*d_Ux6_gpu0,*d_Ux6_gpu2;

    cudaSetDevice(gpu0);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    cudaMalloc((void**)&d_Ux1,sizeof(float)*a*k);
    cudaMalloc((void**)&d_X,sizeof(float)*a*b*c*d);

    cudaMalloc((void**)&d_Ux2_gpu0,sizeof(float)*b*k);
    cudaMalloc((void**)&d_Ux5_gpu0,sizeof(float)*a*b*k);
        
    cudaMalloc((void**)&d_B1,sizeof(float)*k*k*1);
    cudaMalloc((void**)&d_B2,sizeof(float)*k*k*k);

    cudaSetDevice(gpu1);
    cublasHandle_t handle1;
    cublasCreate(&handle1);
    cusolverDnHandle_t cusolverH1 = NULL;
    cusolverDnCreate(&cusolverH1);
    cudaMalloc((void**)&d_X_g1,sizeof(float)*a*b*c*d);
    cudaMalloc((void**)&d_Ux2,sizeof(float)*b*k);

    cudaSetDevice(gpu2);

    cublasHandle_t handle2;
    cublasCreate(&handle2);
    cusolverDnHandle_t cusolverH2 = NULL;
    cusolverDnCreate(&cusolverH2);
    cudaMalloc((void**)&d_X_g2,sizeof(float)*a*b*c*d);
    cudaMalloc((void**)&d_Ux4_gpu2,sizeof(float)*d*k);
    cudaMalloc((void**)&d_Ux6_gpu2,sizeof(float)*c*d*k);
    cudaMalloc((void**)&d_B3,sizeof(float)*k*k*k);

    cudaSetDevice(gpu3);
    cublasHandle_t handle3;
    cublasCreate(&handle3);
    cusolverDnHandle_t cusolverH3 = NULL;
    cusolverDnCreate(&cusolverH3);
     cudaMalloc((void**)&d_X_g3,sizeof(float)*a*b*c*d);
    cudaMalloc((void**)&d_Ux4,sizeof(float)*d*k);

    cudaSetDevice(gpu4);
    cublasHandle_t handle4;
    cublasCreate(&handle4);
    cusolverDnHandle_t cusolverH4 = NULL;
    cusolverDnCreate(&cusolverH4);
    cudaMalloc((void**)&d_X_g4,sizeof(float)*a*b*c*d);
    cudaMalloc((void**)&d_Ux5,sizeof(float)*a*b*k);

    cudaSetDevice(gpu5);
    cublasHandle_t handle5;
    cublasCreate(&handle5);
    cusolverDnHandle_t cusolverH5 = NULL;
    cusolverDnCreate(&cusolverH5);
    cudaMalloc((void**)&d_X_g5,sizeof(float)*a*b*c*d);
    cudaMalloc((void**)&d_Ux6,sizeof(float)*c*d*k);

  cout<<"size :"<<n<<endl;
  gettimeofday(&start,NULL);

  #pragma omp parallel num_threads(6)
  {//  two thread:no.1 to control gpu0 ; no.2 to control gpu1    
    int cpuid = omp_get_thread_num();
    if(cpuid == 0)
    {
      cudaSetDevice(gpu0);
    
      cudaMemcpyAsync(d_X,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);

      gpu0_u1(d_X,a,b,c,d,k,gpu0,d_Ux1,handle,cusolverH);    
      // ttm 计算 B2

    }

    else if(cpuid == 1)
    {
      cudaSetDevice(gpu1);
      
      cudaMemcpyAsync(d_X_g1,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);
      gpu1_u2(d_X_g1,a,b,c,d,k,gpu1,d_Ux2,handle1,cusolverH1);   
           
      cudaMemcpyPeerAsync(d_Ux2_gpu0,gpu0,d_Ux2,gpu1, sizeof(float)*b*k, 0);
    }

    else if(cpuid == 2)
    {
      cudaSetDevice(gpu2);
         
      cudaMemcpyAsync(d_X_g2,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);

      cudaMalloc((void**)&d_Ux3,sizeof(float)*c*k);
      gpu2_u3(d_X_g2,a,b,c,d,k,gpu2,d_Ux3,handle2,cusolverH2);
 
    }

    else if(cpuid == 3)
    {
      cudaSetDevice(gpu3);
           
      cudaMemcpyAsync(d_X_g3,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);
      
      gpu3_u4(d_X_g3,a,b,c,d,k,gpu3,d_Ux4,handle3,cusolverH3);
      
      cudaMemcpyPeerAsync(d_Ux4_gpu2,gpu2,d_Ux4,gpu3, sizeof(float)*d*k, 0);
    }
    else if(cpuid == 4)
    {
      cudaSetDevice(gpu4);     

      
      cudaMemcpyAsync(d_X_g4,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);
      
      gpu4_u5(d_X_g4,a,b,c,d,k,gpu4,d_Ux5,handle4,cusolverH4);
      
      cudaMemcpyPeerAsync(d_Ux5_gpu0,gpu0,d_Ux5,gpu4, sizeof(float)*d*k, 0);
    }

    else if(cpuid == 5)
    {
      cudaSetDevice(gpu5);
            
      cudaMemcpyAsync(d_X_g5,X,sizeof(float)*a*b*c*d,cudaMemcpyHostToDevice,0);
      
      gpu5_u6(d_X_g5,a,b,c,d,k,gpu5,d_Ux6,handle5,cusolverH5);
      
      cudaMemcpyPeerAsync(d_Ux6_gpu2,gpu2,d_Ux6,gpu5, sizeof(float)*d*k, 0);
      cudaMemcpyPeerAsync(d_Ux6_gpu0,gpu0,d_Ux6,gpu5, sizeof(float)*d*k, 0);
    }

    #pragma omp barrier 
   }


    #pragma omp parallel num_threads(2)
  {  
    int cpuid = omp_get_thread_num();

    if(cpuid == 0)
    {
      cudaSetDevice(gpu0);
     
      ttm(d_Ux5_gpu0,d_Ux1,d_Ux2_gpu0,d_B2,a,b,k,k,k,handle);
      ttm(d_X,d_Ux5_gpu0,d_Ux6_gpu0,d_B1,a*b,c*d,1,k,k,handle);
    }
    else if(cpuid == 1)
    {
      cudaSetDevice(gpu2);

      ttm(d_Ux6_gpu2,d_Ux3,d_Ux4_gpu2,d_B3,a,b,k,k,k,handle2);

    }
  }

  gettimeofday(&end,NULL);
  time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
  time_use=time_use/1000000;
  printf("time_use is %.10f\n",time_use);
  // recover
 
  cudaDeviceReset();
  return 0;

}



