
#ifndef GUARD_func_h
#define GUARD_func_h
typedef float dt;

void genHtensor(dt *X,long a,long b,long c,long d);
__global__ void transmission(float *d_A,float *d_B,long a,long b);

void printTensor(float *d_des,long m,long n,long l);
__global__  void floattohalf(dt *AA,half *BB,long m);
void f2h(dt *A,half *B,long num);
__global__ void mode2(float *A,float *B,long m,long n,long r);
__global__ void mode2h(half *A,half *B,long m,long n,long r);

void eig(float *d_A,int m,int n,cusolverDnHandle_t cusolverH);
void QR(float *d_A,int m,int n,cusolverDnHandle_t cusolverH);
void rsvd(float *d_A,float *d_U,int m,int n,int ks,cublasHandle_t handle,cusolverDnHandle_t cusolverH);

void gpu0_u1(float* d_X,int a,int b,int c,int d,int k,int gpu0,float* d_Ux1,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void gpu5_u6(float* d_X,int a,int b,int c,int d,int k,int gpu5,float* d_Ux6,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void gpu4_u5(float* d_X,int a,int b,int c,int d,int k,int gpu4,float* d_Ux5,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void gpu3_u4(float* d_X,int a,int b,int c,int d,int k,int gpu3,float* d_Ux4,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void gpu2_u3(float* d_X,int a,int b,int c,int d,int k,int gpu2,float* d_Ux3,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void gpu1_u2(float* d_X,int a,int b,int c,int d,int k,int gpu1,float* d_Ux2,
          cublasHandle_t handle,cusolverDnHandle_t cusolverH);

void recover(float *d_Ux1,float *d_Ux2,float *d_Ux3,float *d_Ux4,
             float *d_B1,float *d_B2,float *d_B3,float *d_r,
             int a,int b,int c,int d,int k,cublasHandle_t handle);
void ttm(float *d_U1,float *d_U2,float *d_U3,float *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle);
void ttm_tensorcore(half *d_U1,half *d_U2,half *d_U3,float *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle);


#endif