#include "head.h"

int main()
{
	//1200 - 1360

//  规模  1000 ~ 1900

	//cudaSetDevice(2);
	for(long j=1200;j<=1243;j=j+80){ 
		long a=j; //size x
		long b=j;
		long c=j;
		
		cout<<"size:"<<j<<endl;
		
		int *k=new int[j]();
		for(int i=1;i<5;i++)
			k[i]=(int)(j*0.2);
			//k[i]=j;
		k[0]=1;
		//k[1]=2;

		dt *X;
		cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
		//genHtensor(X,a,b,c); //init tensor
		//gentuTensor(X,a,b,c,k[1],k[1],k[1]);
		
		for(long long i = 0;i<a*b*c;i++ ){
			X[i] = rand()*2.0/RAND_MAX - 1.0;
			//X[i] = i+1;
		}
		
		//t1=clock();	
		htd(X,a,b,c,k);		
		//t2=clock();
		//times = (double)(t2-t1)/CLOCKS_PER_SEC;
	
		//cout<<"cost time :"<<times<<"s"<<endl;
		cudaFreeHost(X);
		cudaDeviceReset();
}
	return 0;

}