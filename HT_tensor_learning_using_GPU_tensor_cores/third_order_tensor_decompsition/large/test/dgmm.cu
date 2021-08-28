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
__global__ void sqart_gpu(float *d_A,float *d_B,int m,int n)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
  	const int temp = blockDim.x*gridDim.x;

  	while(i<n*m)
  	{
        int row = i%m;
        int col = i/m;
        if(row == col)
        {
            //d_B[i] = sqrt(d_A[row]);
            d_B[i] = d_A[row];
        }
        else
        {
        	d_B[i]=0;
        }
  		
  		i+=temp;
  	}
  	 __syncthreads();
}
int main()
{
	int a=3;
	int b=2;

	float *A = new float[b*b];
	float *S = new float[b];
	for(unsigned i = 0; i < a*b; ++i) {
		/* code */
		A[i] = i+1;
	}
	for(unsigned i = 0; i < b; ++i) {
		/* code */
		S[i] = i+1;
	}

	float *d_A,*d_S,*d_W;
	cudaMalloc((void**)&d_A,sizeof(float)*a*b);
	cudaMalloc((void**)&d_S,sizeof(float)*b);
	//cudaMalloc((void**)&d_W,sizeof(float)*b*b);
	//cudaMemcpy(d_A,A,sizeof(float)*b*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_S,S,sizeof(float)*b,cudaMemcpyHostToDevice);

	sqart_gpu<<<512,512>>>(d_S,d_A,a,b);

	printTensor(d_A,a,b,1);




}