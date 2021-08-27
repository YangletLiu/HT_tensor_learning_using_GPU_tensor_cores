#include<iostream>
#include<fstream>
#include <assert.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda_fp16.h>
using namespace std;
void printTensor(float *d_des,long m,long n,long l){
    float *des = new float[m*n*l]();
    cudaMemcpy(des,d_des,sizeof(float)*m*n*l,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int k = 0;k<l;k++){
        for(int i = 0;i<n;i++){
            for(int j = 0;j<m;j++){
                cout<<des[k*m*n+i*m+j]<<" ";
            }
            cout<<endl;
        }
        cout<<"~~~~~~~~~~~~~~~~"<<endl;
    }
    delete[] des;des=nullptr;

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
__global__ void abs_kernel(float *d_A,int a,int b)
{
    long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;
    while(i<a*b)
    {
        d_A[i] = fabs(d_A[i]);
        i+=temp;
    }
    __syncthreads();
}

void svd(float *d_A,float *d_U,int a,int b,int k,cublasHandle_t handle,cusolverDnHandle_t cusolverH)   // a > b  降序
{

    float *d_S = NULL;   
    float *d_VT = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_rwork = NULL;
    float *d_W = NULL;  // W = S*VT

    int lwork = 0;
    int info_gpu = 0;

    cudaMalloc ((void**)&d_S  , sizeof(float)*b);
    cudaMalloc ((void**)&d_VT , sizeof(float)*a*b);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaMalloc ((void**)&d_W  , sizeof(float)*a*b);

    cusolverDnSgesvd_bufferSize(cusolverH,a,b,&lwork );
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
     cusolverDnSgesvd (
        cusolverH,jobu,jobvt,a,b,d_A,a,d_S,d_U,a,  // ldu
        d_VT,
        a, // ldvt,
        d_work,lwork,d_rwork,devInfo);
    cudaDeviceSynchronize();
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("after gesvd: info_gpu = %d\n", info_gpu); 

    cublasScopy(handle,n*k,d_U,1,d_Rk,1);

    if (d_S    ) cudaFree(d_S);    
    if (d_VT   ) cudaFree(d_VT);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
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
    upper<<<threads,block0>>>(d_A,d_R,m,n); // 获得R
    cudaDeviceSynchronize();
    cusolverDnSorgqr(cusolverH,m,n,n,d_A,m,d_tau,d_work2,lwork2,devInfo2); //获得 Q

    cudaFree(d_tau);
    cudaFree(devInfo2);
    cudaFree(d_work2);
}


int main()
{
    //int m =100*100;
    int m = 20;
    int n = 4;
    int k=0.5*n;

    float *A,*d_A;
    cudaHostAlloc((void**)&A,sizeof(float)*m*n, 0);
    cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    printf("init data\n");
    for(long long  i = 0; i < m*n; ++i) {
        A[i] = i+1;
        //A[i]= rand()*0.1/(RAND_MAX*0.1);
    }
    cudaMemcpy(d_A,A,sizeof(float)*m*n,cudaMemcpyHostToDevice);
    int p=4;//将A按照垂直方向分为p块，每块size (m/4)*n
    int d = m/4;

    float *B[4];
    cudaHostAlloc((void**)&B[0],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[1],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[2],sizeof(float)*d*n,0);
    cudaHostAlloc((void**)&B[3],sizeof(float)*d*n,0);

    int row,col;
    for(long i = 0; i < m*n; ++i) {
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

    cublasHandle_t handle;
    cublasCreate(&handle);
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    float alpha = 1.0;
    float beta = 0.0;

    float *d_B[4];
    for(int i=0;i<p;i++)
    {
        cudaMalloc((void**)&d_B[i],sizeof(float)*d*n);
        cudaMemcpyAsync(d_B[i],B[i],sizeof(float)*d*n,cudaMemcpyHostToDevice,0);
    }
    float *d_R[4];
   // printTensor(d_B[0],d,n,1);
    for(int i =0;i<p;i++)
    {
        cudaMalloc((void**)&d_R[i],sizeof(float)*n*n);  
        QR(d_B[i],d,n,d_R[i],handle,cusolverH);
        cudaDeviceSynchronize();

        //printTensor(d_B[i],d,n,1); 这里的结果与 matlab QR分解后的结果不同
    }   
    //printTensor(d_R[0],4,4,1);
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

    //printTensor(d_R[0],n,n,1);//这里没问题

    //获得矩阵Q
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

    //d_B[i] * d_Ri.T  因此d_Ri 是通过转置之后得到的，因此乘法时要转置,结果仍存放在d_B[i]中
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[0],d,d_R1,n,&beta,d_B[0],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[1],d,d_R2,n,&beta,d_B[1],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[2],d,d_R3,n,&beta,d_B[2],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[3],d,d_R4,n,&beta,d_B[3],d);

    //上面的结果再次与d_R1234T的Q 相乘
    //d_R1234T同样垂直方向分为两个矩阵 
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,2*n,&alpha,d_R1234T,2*n,&beta,d_R1234T,2*n,d_R1234,n);
    cublasScopy(handle,n*n,d_R1234,1,d_R1T,1);
    cublasScopy(handle,n*n,d_R1234+n*n,1,d_R2T,1);

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[0],d,d_R1T,n,&beta,d_B[0],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[1],d,d_R1T,n,&beta,d_B[1],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[2],d,d_R2T,n,&beta,d_B[2],d);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,d,n,n,&alpha,d_B[3],d,d_R2T,n,&beta,d_B[3],d);
    //d_B[i] 垂直叠放为所需的Q

    //2、对d_R[0],即求得最后的R(n*n) svd 分解,结果保存在d_R中
   // printTensor(d_R[0],n,n,1);
    svd(d_R[0],d_R[1],n,n,handle,cusolverH);

   // printTensor(d_R[1],n,n,1); //z这里也不一样

    //3、 U = Q * U
    for(unsigned i = 0; i < p; ++i) {
        /* code */
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,n,n,&alpha,d_B[i],d,d_R[1],n,&beta,d_B[i],d);
    }
    float *d_tmp,*d_UT,*d_U;
    cudaMalloc((void**)&d_tmp,sizeof(float)*d*n);
    cudaMalloc((void**)&d_UT,sizeof(float)*m*n);
    cudaMalloc((void**)&d_U,sizeof(float)*m*n);
    for(unsigned i = 0; i < p; ++i) {
        /* code */
        cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,n,d,&alpha,d_B[i],d,&beta,d_B[i],d,d_tmp,n);
        cublasScopy(handle,n*d,d_tmp,1,d_UT+i*d*n,1);
    }
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,&alpha,d_UT,n,&beta,d_UT,n,d_U,m);
 
    //printTensor(d_U,16,4,1);


    // 下面是直接svd,然后用求出的S V与上面的U相乘，反向求出原始矩阵，与原矩阵比较
    float *d_Ureal;
    cudaMalloc((void**)&d_Ureal,sizeof(float)*m*m);

    float *d_S = NULL; 
    float *d_W=NULL;  
    float *d_VT = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_rwork = NULL;
    float *d_A2;

    int lwork = 0;
    int info_gpu = 0;
    cudaMalloc((void**)&d_A2,sizeof(float)*m*n);
    cudaMalloc ((void**)&d_W  , sizeof(double)*m*n);
    cudaMalloc ((void**)&d_S  , sizeof(float)*n);
    cudaMalloc ((void**)&d_VT , sizeof(float)*m*n);
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cusolverDnSgesvd_bufferSize(cusolverH,m,n,&lwork );
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
     cusolverDnSgesvd (
        cusolverH,jobu,jobvt,m,n,d_A,m,d_S,d_Ureal,m,  // ldu
        d_VT,
        m, // ldvt,
        d_work,lwork,d_rwork,devInfo);
    cudaDeviceSynchronize();

    cublasSdgmm(
        handle,
        CUBLAS_SIDE_LEFT,
        n,
        n,
        d_VT,
        m,
        d_S,
         1,
        d_W,
        m);
    cublasSgemm(
        handle,
        CUBLAS_OP_N, // U
        CUBLAS_OP_N, // W
        m, // number of rows of A
        n, // number of columns of A
        n, // number of columns of U 
        &alpha, /* host pointer */
        d_U, // U
        m,
        d_W, // W
        m,
        &beta, /* hostpointer */
        d_A2,
        m);
    //printTensor(d_A2,m,n,1);
    float alpha1=-1.0;
    float re=0.0;
    float before = 0.0;

    cublasSaxpy(handle,m*n, &alpha1,d_A,1,d_A2,1);
    cublasSnrm2(handle,m*n,d_A2,1,&re);
    cublasSnrm2(handle,m*n,d_A,1,&before);
    cudaDeviceSynchronize();
    cout<<"error rate "<<re/before<<endl;  

    cudaFree(d_R12);
    cudaFree(d_R34);
    cudaFree(d_R1T);
    cudaFree(d_R2T);

    // 结果，在16*4这样的矩阵中，出现问题是 上面的8*4精度没问题，下面的8*4精度很差
    //matlab 证实在较大规模时精度还可以，并且同样的数值，这里出现问题，应该发生程序错误

}