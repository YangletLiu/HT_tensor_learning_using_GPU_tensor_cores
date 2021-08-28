#include "head.h"
#include<unistd.h>
#include "nvml.h"

int main()
{

	clock_t t1,t2;
	double times=0.0;
	ofstream fout("time.txt",ios::app);
	for(long j=168;j<=169;j=j+16){
		long a=j; //size x
		long b=j;
		long c=j;
		long d=j;
		
		cout<<"size:"<<j<<endl;
	
		int *k=new int[7]();
		for(int i=1;i<7;i++)
			k[i]=(int)(j*0.1);
			//k[i]=j;
		k[0]=1;
		//k[1]=2;

		dt *X;
		cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c*d,0);
		genHtensor(X,a,b,c,d); //生产一个低秩tensor(通过hosvd来生成)	


		nvmlReturn_t result;
		result = nvmlInit();
		int device_count;
		cudaGetDevice(&device_count);
		nvmlDevice_t device;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlPciInfo_t pci;
        result = nvmlDeviceGetHandleByIndex(device_count, &device);
        if (NVML_SUCCESS != result) {
            std::cout << "get device failed " << endl;
        }
        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result) {
            std::cout << "GPU name： " << name << endl;
        }

		cublasHandle_t handle;
		cublasCreate(&handle);
		cusolverDnHandle_t cusolverH = NULL;
		cusolverDnCreate(&cusolverH);

		t1=clock();	
		htd4(X,a,b,c,d,k,handle,cusolverH);
		
		t2=clock();
		times = (double)(t2-t1)/CLOCKS_PER_SEC;
		
		nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (NVML_SUCCESS == result)
        {
            std::cout << " device :"<< device_count <<"utilize.";
            std::cout << " GPU utilize： " << utilization.gpu << " device memory utilize " << utilization.memory << endl;
        }else{
        	cout<<"fail ！"<<endl;
        }




		cout<<"cost time :"<<times<<"s"<<endl;
		fout<<times<<"\r\n";
		cudaFreeHost(X);
		cudaDeviceReset();
}
fout.close();
	return 0;

}