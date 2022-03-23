
#ifndef GUARD_func_h
#define GUARD_func_h
typedef float dt;

void htd4(dt *x,long a,long b,long c,long d,int *k,cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void ttm(dt *d_U1,dt *d_U2,dt *d_U3,dt *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle);
void ttm_tensorcore(half *d_U1,half *d_U2,half *d_U3,dt *d_B,int a,int b,int k1,int k2,int k3,cublasHandle_t handle);
void printTensor(dt *d_des,long m,long n,long l,long k);
__global__ void mode2(dt *A,dt *B,long m,long n,long r);

void genHtensor(dt *X,long a,long b,long c,long d);
void QR(float *d_A,int m,int n,cusolverDnHandle_t cusolverH);
void svd(float *d_B,int m,int n,float *d_UT,float *d_S,float *d_V,cublasHandle_t cublasH,cusolverDnHandle_t cusolverH);
void rsvd(float *d_A,float *d_U,int m,int n,int ks,cublasHandle_t handle,cusolverDnHandle_t cusolverH);
void eig(float *d_A,int m,int n,cusolverDnHandle_t cusolverH);
__global__ void upper(dt *A,dt *R,int m,int n);
__global__ void initIdeMat(dt *AA,int m);
__global__ void transmission(dt *d_A,dt *d_B,long a,long b);

void f2h(dt *A,half *B,long num);
__global__ void mode2h(half *A,half *B,long m,long n,long r);
__global__  void floattohalf(dt *AA,half *BB,long m);

#endif