#include "head.h"

int main()
{

	clock_t t1,t2;
	double times=0.0;
	ofstream fout("time.txt",ios::app);
	for(long j=24;j<=168;j=j+16){
		long a=j; //size x
		long b=j;
		long c=j;
		long d=j;
		
		cout<<"size:"<<j<<endl;
	
		int *k=new int[7]();
		for(int i=1;i<7;i++)
			k[i]=(int)(j*0.5);
			//k[i]=j;
		k[0]=1;
		//k[1]=2;

		dt *X;
		cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c*d,0);
		genHtensor(X,a,b,c,d); //生产一个低秩tensor(通过hosvd来生成)

		t1=clock();	
		htd4(X,a,b,c,d,k);
		
		t2=clock();
		times = (double)(t2-t1)/CLOCKS_PER_SEC;
		
		cout<<"cost time :"<<times<<"s"<<endl;
		fout<<times<<"\r\n";
		cudaFreeHost(X);
		cudaDeviceReset();
}
fout.close();
	return 0;

}