#include "head.h"
#include<unistd.h>
#include "nvml.h"

int main()
{

	
	for(long j=1120;j<=1140;j=j+80){
		long a=j; //size x
		long b=j;
		long c=j;
		
		cout<<"size:"<<j<<endl;
		
		int *k=new int[j]();
		for(int i=1;i<5;i++)
			k[i]=(int)(j*0.1);
			//k[i]=j;
		k[0]=1;
		//k[1]=2;

		dt *X;
		float error;
		cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
		genHtensor(X,a,b,c); //init tensor
		//gentuTensor1(X,a,b,c,k[1],k[2],k[3]);
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
        

		error = htd(X,a,b,c,k);

		    //使用率
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (NVML_SUCCESS == result)
        {
            std::cout << " device :"<< device_count <<"utilize.";
            std::cout << " GPU utilize： " << utilization.gpu << " device memory utilize " << utilization.memory << endl;
        }else{
        	cout<<"fail ！"<<endl;
        }
		



		//htd_tensor(X,a,b,c,k);

		cudaFreeHost(X);
		cudaDeviceReset();
}
	return 0;

}