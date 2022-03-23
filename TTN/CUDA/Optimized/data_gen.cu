#include "head.h"

void random_isometry(float *iso,int m,int n) 
{
	
	float *d_A;  //需要生成的A 大小为 n*m  不是 m*n
	cudaMalloc((void**)&d_A,sizeof(float)*m*n);
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1233ULL);
	curandGenerateNormal(gen,d_A,m*n,0,1);
	// 对d_A QR分解，然后得到 Q，R
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	int *devInfo2 = NULL;
    float *d_work2 = NULL;
    int  lwork_geqrf = 0;
    int  lwork_orgqr = 0;
    int  lwork2 = 0;
    float *d_R;
    float *d_tau = NULL;
    cudaMalloc((void**)&d_R,sizeof(float)*m*m);     
    cudaMalloc ((void**)&d_tau, sizeof(float) * m);
    cudaMalloc ((void**)&devInfo2, sizeof(int));

    cusolverDnSgeqrf_bufferSize(cusolverH,n,m,d_A,n,&lwork_geqrf);
    cusolverDnSorgqr_bufferSize(cusolverH,n,m,m,d_A,n, d_tau,&lwork_orgqr);

    lwork2 = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
    cudaMalloc((void**)&d_work2, sizeof(float)*lwork2);

    cusolverDnSgeqrf(cusolverH,n,m,d_A,n,d_tau,d_work2,lwork2,devInfo2);
    upper<<<512,512>>>(d_A,d_R,n,m); // 获得R
    cudaDeviceSynchronize();
    cusolverDnSorgqr(cusolverH,n,m,m,d_A,n,d_tau,d_work2,lwork2,devInfo2);
    //d_A 是Q
    float *d_r;
    cudaMalloc((void**)&d_r,sizeof(float)*m);

    dim3 block0((m*m+1024-1)/1024,1,1);
    diag_part<<<block0,1024>>>(d_R,d_r,m);

    R_div_r<<<block0,1024>>>(d_R,d_r,m);

    cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0;
	float beta = 0.0;

	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,m,&alpha,d_R,m,d_A,n,&beta,iso,m);
	cudaDeviceSynchronize();
	cudaFree(d_A);
	cudaFree(d_R);
	cudaFree(d_tau);
	cudaFree(d_r);
	cudaFree(devInfo2);
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);

}
void reshape(float *iso,int a,int b,int c)
{	
	//b=c
	float *d_AT,*d_tmp;
	cudaMalloc((void**)&d_AT,sizeof(float)*a*b*c);
	cudaMalloc((void**)&d_tmp,sizeof(float)*b*b);

	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0;
	float beta = 0.0;

	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,b*c,a,&alpha,iso,a,&beta,iso,a,d_AT,b*c);
	dim3 block0((b*b+1024-1)/1024,1,1);
	initIdeMat<<<block0,1024>>>(d_tmp,b);

	cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_T, b,b,b,&alpha,d_AT, b,b*b,d_tmp, b,0,
                                  &beta,iso, b*b,b,a);
	cudaDeviceSynchronize();

	cudaFree(d_AT);
	cudaFree(d_tmp);
	cublasDestroy(handle);
}

void random_tree_tn_uniform(int *Ds,float **iso,int **sizeiso)
{
	int *Ds2 = new int[num_layers+1];
	memcpy(Ds2,Ds,sizeof(int)*num_layers);
	Ds2[num_layers] = 1;//top_ranks
	int tmp;
	cout<<" "<<endl;
	for(int j=0;j<num_layers;j++)
	{	
		cudaHostAlloc((void**)&sizeiso[j],sizeof(int)*3,0);
		tmp = pow(Ds2[j],2);
		if(Ds2[j+1] == tmp)
		{
			//iso[j]=一个对角阵
			cudaMalloc((void**)&iso[j],sizeof(float)*Ds2[j+1]*Ds2[j+1]);
			dim3 block0((Ds2[j+1]*Ds2[j+1]+1024-1)/1024,1,1);
			eye<<<block0,1024>>>(iso[j],Ds2[j+1]);
		}
		else{
			//随机生成
			cudaMalloc((void**)&iso[j],sizeof(float)*Ds2[j+1]*tmp);
			random_isometry(iso[j],Ds2[j+1],tmp);
		}
		//reshape(Ds2[j+1],Ds2[j],Ds2[j]) 每列作为一个矩阵，并转置
		
		reshape(iso[j],Ds2[j+1],Ds2[j],Ds2[j]); //reshape之后的size为Ds2[j] * Ds2[j] * Ds2[j+1]
		//这里获得每个iso 的shape
		sizeiso[j][0] = Ds2[j];sizeiso[j][1] = Ds2[j];sizeiso[j][2] = Ds2[j+1];
		//cout<<sizeiso[j][0]<<" "<<sizeiso[j][1]<<" "<<sizeiso[j][2]<<endl;  
	}
	//printTensor(iso[1],Ds2[1],Ds2[1],Ds2[2]);
}