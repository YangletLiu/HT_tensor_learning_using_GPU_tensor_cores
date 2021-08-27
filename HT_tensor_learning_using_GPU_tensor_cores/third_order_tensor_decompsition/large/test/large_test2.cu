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
typedef float dt;
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
void  gentuTensor1(float *X,long a,long b,long c,long r1,long r2,long r3)
{
    dt *A,*B,*C,*G;
    cudaHostAlloc((void**)&A,sizeof(dt)*a*r1,0);
    cudaHostAlloc((void**)&B, sizeof(dt)*b*r2,0);
    cudaHostAlloc((void**)&C, sizeof(dt)*c*r3,0);
    cudaHostAlloc((void**)&G, sizeof(dt)*r1*r2*r3,0);
    srand(123);

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




    dt * d_A,*d_B,*d_C,*d_X,*d_G;
    cudaMalloc((void**)&d_A,sizeof(dt)*a*r1);
    cudaMalloc((void**)&d_B,sizeof(dt)*b*r2);
    cudaMalloc((void**)&d_C,sizeof(dt)*c*r3);
    cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);
    cudaMalloc((void**)&d_G,sizeof(dt)*r2*r1*r3);

    cudaMemcpyAsync(d_A, A,sizeof(dt)*a*r1,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_B, B,sizeof(dt)*b*r2,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_C, C,sizeof(dt)*c*r3,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_G, G,sizeof(dt)*r1*r2*r3,cudaMemcpyHostToDevice,0);
    dt *d_AG,*d_AGB;
    cudaMalloc((void**)&d_AG,sizeof(dt)*a*r2*r3);
    cudaMalloc((void**)&d_AGB,sizeof(dt)*b*a*r3);

    // X(1) = A*G(1)*(C Kronecker B)
    // A a*r1  G(1) r1*(r2*r3)  
    dt alpha = 1.0;
    dt beta =0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                a,r2*r3,r1,
                &alpha,
                d_A,a,
                d_G,r1,
                &beta,d_AG,a
                );
    //cout<<"AG"<<endl; printTensor(d_AG,3,2,1);
    // AG * B(T)  a*r2*r3   (b*r2)T

        cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_T,
                                  a,b,r2,
                                  &alpha,
                                  d_AG,a,a*r2,
                                  d_B,b,0,
                                  &beta,
                                  d_AGB,a,a*b,r3
                                  );
    
    
//cout<<"CKRB"<<endl; printTensor(d_CkrB,3,2,1);
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                a*b,c,r3,
                &alpha,
                d_AGB,a*b,
                d_C,c,
                &beta,
                d_X,a*b
                );

    cudaMemcpyAsync(X,d_X,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,  0);
    cudaDeviceSynchronize();
    //cout<<"X"<<endl; printTensor(d_X,3,2,1);

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


__global__ void sqrt_gpu2(float *d_A,float *d_B,int b,int k)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    const int temp = blockDim.x*gridDim.x;

    while(i<k*k)
    {
        int row = i%k;
        int col = i/k;
        if(row == col)
        {
            d_B[i] = sqrt(d_A[b-k+row]);                    
        }
        else
        {
            d_B[i]=0;
        }       
        i+=temp;
    }
     __syncthreads();
}

__global__ void initIdeMat(float *AA,int m){
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
__global__ void abs_kernel(float *d_A,int a,int b,int c)
{
    long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;
    while(i<a*b*c)
    {
        d_A[i] = fabs(d_A[i]);
        i+=temp;
    }
    __syncthreads();
}

int main()
{
	int a = 200;
    int b = 200;
    int c = 200;
    int k = 20;

    int p = 5;  //分为 p 片来传输
    int slice = c/p;

  

    float *A = new float[a*b*c];
    float *A_mode3 = new float[a*b*c];

    gentuTensor1(A,a,b,c,k,k,k);
   /* for(long i = 0; i < a*b*c; ++i) {        
        A[i]=rand()*0.1/(RAND_MAX);
    }*/

     for(int i=0;i<a*b;i++)
    {
        for(int j = 0; j < c; ++j) {
            A_mode3[i*c+j] = A[i+j*a*b];          
        }
    }

    float *d_A;
    cudaMalloc((void**)&d_A,sizeof(float)*a*b*c);
    cudaMemcpy(d_A,A,sizeof(float)*a*b*c,cudaMemcpyHostToDevice);


	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);

	float alpha = 1.0;
	float beta = 0.0;
	dim3 threads0(512,1,1);
	dim3 block1((slice*slice+512-1)/512,1,1); //for X3
    
    float *d_X3,*d_X3_X3,*d_X3T,*d_Xtemp,*d_Xtemp1;
    cudaMalloc((void**)&d_X3,sizeof(float)*c*c);
    cudaMalloc((void**)&d_X3T,sizeof(float)*c*slice);
    cudaMalloc((void**)&d_X3_X3,sizeof(float)*c*c);
	cudaMalloc((void**)&d_Xtemp,sizeof(float)*a*b*slice);
	cudaMalloc((void**)&d_Xtemp1,sizeof(float)*a*b*slice);

	float *d_Idemat3;
	cudaMalloc((void**)&d_Idemat3,sizeof(float)*slice*slice);
	initIdeMat<<<block1,threads0>>>(d_Idemat3,slice);


    for(int i = 0;i<p;i++){
		cudaMemcpyAsync(d_Xtemp,A+i*a*b*slice,sizeof(float)*a*b*slice,cudaMemcpyHostToDevice,0);		
		
		for (int j = 0;j<p;j++){
			cudaMemcpyAsync(d_Xtemp1,A+j*a*b*slice,sizeof(float)*a*b*slice,cudaMemcpyHostToDevice,0);
			cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,slice,slice,a*b,&alpha,d_Xtemp1,a*b,d_Xtemp,a*b,&beta,d_X3+(i*p+j)*slice*slice,slice);
		//	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,a*b,&alpha,d_Xtemp,1,d_Xtemp1,a*b,&beta,d_X3_X3+i*c+j,1);
		}// d_X3 is size of slice *c transpose to c*slice
		cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,slice,slice,slice,&alpha,d_X3+i*c*slice,slice,slice*slice,d_Idemat3,slice,0,&beta,d_X3T,slice,slice*slice,p);
		cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,slice,&alpha,d_X3T,slice,&beta,d_X3_X3+i*c*slice,c,d_X3_X3+i*c*slice,c);
	}

	//printTensor(d_X3_X3,4,4,1);

	//对 d_X3_X3 特征分解  求得 S V   (c*c)
	float *d_W = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    int  lwork = 0;

    cudaMalloc ((void**)&d_W, sizeof(float) * c);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnSsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        c,
        d_X3_X3,
        c,
        d_W,
        &lwork);
     cudaMalloc((void**)&d_work, sizeof(float)*lwork);
     cusolverDnSsyevd(
        cusolverH,jobz,uplo,c,d_X3_X3,c,
        d_W,d_work,lwork,devInfo);


    dim3 threads(1024,1,1);
    dim3 block0((k*k+1024-1)/1024,1,1); //for X2
    float *d_S,*d_ST,*d_AK;      
    cudaMalloc((void**)&d_S,sizeof(float)*k*k);
    cudaMalloc((void**)&d_ST,sizeof(float)*c*k);
    cudaMalloc((void**)&d_AK,sizeof(float)*c*k); //c行k列


    sqrt_gpu2<<<threads,block0>>>(d_W,d_S,c,k);
    cublasScopy(handle,b*k,d_X3_X3+c*(c-k),1,d_AK,1);

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,c,k,k,&alpha,d_AK,c,d_S,k,&beta,d_ST,c);

    float *d_SV;
     cudaMalloc((void**)&d_SV,sizeof(float)*c*k);
     cublasScopy(handle,c*k,d_ST,1,d_SV,1);



    //现对ST QR 分解得到 Q和R  c*k --->  Q c*k  R k*k
    int *devInfo2 = NULL;
    float *d_work2 = NULL;
    int  lwork_geqrf = 0;
    int  lwork_orgqr = 0;
    int  lwork2 = 0;
    float *d_R;
    float *d_tau = NULL;
    cudaMalloc((void**)&d_R,sizeof(float)*k*k);     
    cudaMalloc ((void**)&d_tau, sizeof(float) * k);
    cudaMalloc ((void**)&devInfo2, sizeof(int));

    cusolverDnSgeqrf_bufferSize(cusolverH,c,k,d_ST,c,&lwork_geqrf);
    cusolverDnSorgqr_bufferSize(cusolverH,c,k,k,d_ST,c, d_tau,&lwork_orgqr);

    lwork2 = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
    cudaMalloc((void**)&d_work2, sizeof(float)*lwork2);

    cusolverDnSgeqrf(cusolverH,c,k,d_ST,c,d_tau,d_work2,lwork2,devInfo2);
    upper<<<threads,block0>>>(d_ST,d_R,c,k); // 获得R
    cudaDeviceSynchronize();
    cusolverDnSorgqr(cusolverH,c,k,k,d_ST,c,d_tau,d_work2,lwork2,devInfo2); //获得 Q
   // printTensor(d_ST,4,4,1);

    //接下来分片，一片一片来求 U.T  k*(a*b) 

    //int p=10;
    int slice2 = (a*b)/p;


    float *d_B,*d_Amode3,*d_tempB;
    cudaMalloc((void**)&d_B,sizeof(float)*k*a*b);
    cudaMalloc((void**)&d_Amode3,sizeof(float)*c*slice2);
    cudaMalloc((void**)&d_tempB,sizeof(float)*k*slice2);

    for(int i = 0; i < p; ++i) {

        cudaMemcpy(d_Amode3,A_mode3+i*c*slice2,sizeof(float)*c*slice2,cudaMemcpyHostToDevice);
        //printTensor(d_Amode3,4,4,1);
        
        cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k,slice2,c,
                &alpha,d_ST,c,d_Amode3,c,
                &beta,d_tempB,k
                );
   
        cublasStrsm(   // d_R 运算前后是不会变的
         handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
         k,slice2,
         &alpha,d_R,k,
         d_tempB,k);
        
        cublasScopy(handle,k*slice2,d_tempB,1,d_B+i*k*slice2,1);
        cudaDeviceSynchronize();
    }

    float *d_Ux2;
    cudaMalloc((void**)&d_Ux2,sizeof(float)*a*b*k);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,a*b,k,&alpha,d_B,k,&beta,d_B,k,d_Ux2,a*b);



   //SVD 分解A，得到 U来计算损失

    int *devInfo3 = NULL;
    float *d_work3 = NULL;
    float *d_rwork3 = NULL;
    int lwork3 = 0;
    int info_gpu3 = 0;
    float *d_S2 = NULL;
    float *d_U2 = NULL;
    float *d_VT = NULL;

    cudaMalloc ((void**)&d_S2  , sizeof(float)*c);
    cudaMalloc ((void**)&d_U2  , sizeof(float)*a*b*a*b);
    cudaMalloc ((void**)&d_VT , sizeof(float)*a*b*c);
    cudaMalloc ((void**)&devInfo3, sizeof(int));

    cusolverDnSgesvd_bufferSize(cusolverH,a*b,c,&lwork3 );
    cudaMalloc((void**)&d_work3 , sizeof(float)*lwork3);
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT

    //printTensor(d_A,4,4,1);

    cusolverDnSgesvd (cusolverH,jobu,jobvt,
        a*b,c,d_A,a*b,
        d_S2,
        d_U2,
        a*b,  // ldu
        d_VT,
        a*b, // ldvt,
        d_work3,
        lwork3,
        d_rwork3,
        devInfo3);

    dim3 block3((a*b*c+1024-1)/1024,1,1);
    float *d_Ux2_t,*d_Ux2_r;
    cudaMalloc((void**)&d_Ux2_t,sizeof(float)*a*b*c);
    cudaMalloc((void**)&d_Ux2_r,sizeof(float)*a*b*k);


    transmission<<<block3,threads>>>(d_U2,d_Ux2_t,a*b,c);
    cublasScopy(handle,a*b*k,d_Ux2_t+(c-k)*a*b,1,d_Ux2_r,1);

    dim3 threadsk(1024,1,1);
    dim3 blockk((a*b*k+1024-1)/1024,1,1);

    abs_kernel<<<threadsk,blockk>>>(d_Ux2,a,b,k);
    abs_kernel<<<threadsk,blockk>>>(d_Ux2_r,a,b,k);



    printTensor(d_Ux2,4,4,1);
    printTensor(d_Ux2_r,4,4,1);



    float alpha1=-1.0;
    float re=0.0;
    float before = 0.0;

    cublasSaxpy(handle,a*b*k,&alpha1,d_Ux2_r,1,d_Ux2,1);
    //printTensor(d_Ux2,a,b,k); 
    cublasSnrm2(handle,a*b*k,d_Ux2,1,&re);
    cublasSnrm2(handle,a*b*k,d_Ux2_r,1,&before);
    cudaDeviceSynchronize();
    cout<<"error rate "<<re/before<<endl;  

    //结果，这里也保证了没有相反数(全部取绝对值),在(200*200)*200的时候，误差0.4

}

