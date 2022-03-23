#include "head.h"

int main()
{

	dt rel_eps=1e-7/2;
	cudaSetDevice(2);
	for(long j=80;j<=81;j=j+80){ 
		long a=j; //size x
		long b=j;
		long c=j;
		cout<<"size:"<<j<<endl;
		dt max_rank=j;
		int *k=new int[j]();
		for(int i=1;i<5;i++)
			k[i]=(int)(j*0.5);
		k[0]=1;

	
		dt *X;
		cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
		genHtensor(X,a,b,c); //init tensor
		float error;
		
		error = htd(X,a,b,c,k,rel_eps,max_rank);
		cudaFreeHost(X);
		cudaDeviceReset();
}
	return 0;

}