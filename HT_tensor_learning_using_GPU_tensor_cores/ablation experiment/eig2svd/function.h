
#ifndef GUARD_func_h
#define GUARD_func_h
typedef float dt;

float htd(dt *x,long a,long b,long c,int *k,dt rel_eps,dt max_rank);
void printTensor(dt *d_des,long m,long n,long l);
void genHtensor(dt *X,long a,long b,long c);
void qr_svd(dt *d_A,dt *d_U,int a,int b);
void qr_svd_2(dt *d_A,dt *d_U,int a,int b);
void gesvda(dt *d_A,dt *d_U,int a,int b,int k);
// void svd_VT(float *d_A,float *d_VT,int a,int b,cublasHandle_t handle);
// void gesvdj(float *d_AT,float *d_V,int b,int a);
__global__ void upper(dt *A,dt *R,int m,int n);
__global__ void sqrt_T(dt *A,dt *B,int a);
__global__ void upper_1(dt *R,int n);
__global__ void norm_sum(dt *A,dt *B,int a);

void printvec(dt *d_des,long m,long n,long l);
__global__ void truncate_h(dt *d_A,dt *d_B,long a,long b);
__global__ void transmission(dt *d_A,dt *d_B,long a,long b);
__global__ void tensorToMode1(dt *T1,dt *T2,int m,int n,int k );
__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k);
__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k);
void evd(float *d_A,int m,cublasHandle_t handle,cusolverDnHandle_t cusolverH);
__global__ void transmission_for_svd(dt *d_A,dt *d_B,long a,long b);
#endif