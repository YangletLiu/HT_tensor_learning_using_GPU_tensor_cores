
#ifndef GUARD_func_h
#define GUARD_func_h
typedef float dt;

void htd(dt *x,long a,long b,long c,int *k);
void printTensor(dt *d_des,long m,long n,long l);
__global__ void initIdeMat(dt *AA,int m);
__global__ void initIdeMat_h(half *AA,int m);
void genHtensor(dt *X,long a,long b,long c);
void  gentuTensor(float *X,long a,long b,long c,long r1,long r2,long r3);
void eig(float *A,int m,int n,cusolverDnHandle_t cusolverH);

void QR(float *d_A,int m,int n,float *d_R,cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void gesvda(dt *d_A,dt *d_U,int a,int b,int k);
void svd(float *d_A,float *d_U,int a,int b,cusolverDnHandle_t cusolverH);
void tsqr_svd(float *A,int a,int b,int c,int k,float *d_U,cublasHandle_t handle, cusolverDnHandle_t cusolverH);
void tsqr_svd_half(float *A,int a,int b,int c,int k,float *d_U,cublasHandle_t handle, cusolverDnHandle_t cusolverH);
__global__ void upper(dt *A,dt *R,int m,int n);
__global__ void mode2(dt *A,dt *B,long m,long n,long r);
__global__ void transmission(dt *d_A,dt *d_B,long a,long b);
__global__ void upper(float *A,float *R,int m,int n);
__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k);
void f2h(dt *A,half *B,long num);
__global__  void floattohalf(dt *AA,half *BB,long m);
void ttm(dt *d_U1,dt *d_U2,dt *d_U3,dt *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle);
void ttm_tensorcore(half *d_U1,half *d_U2,half *d_U3,dt *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle);

#endif