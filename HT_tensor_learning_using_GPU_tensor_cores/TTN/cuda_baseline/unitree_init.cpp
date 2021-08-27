#include "head.h"


int main()
{	

	int max_bond_dim = 16;
	int build_graphs = 0;
	int verbose = 1;
	int num_sweeps = 10;

	int temp;
	int *Ds = new int[num_layers];
	for(int i = 1; i < num_layers+1; ++i) {
		temp = pow(2,i);
		if(temp > max_bond_dim)
		{
			temp = max_bond_dim;
		}
		Ds[i-1] = temp;
	}
	float h1[4]={-1,0,0,1};
	//h1[0]=-1;h1[1]=0;h1[2]=0;h1[3]=1;
	float h2[4];
	h2[0]=0;h2[1]=-1;h2[2]=-1;h2[3]=0;
	h2_mpo h_mpo_2site;
	h_mpo_2site.init1(2,2);
	h_mpo_2site.init2(2,2);	

	cudaMemcpy(h_mpo_2site.mpo1,h2,sizeof(float)*4,cudaMemcpyHostToDevice);

	h2[0]=0;h2[1]=1;h2[2]=1;h2[3]=0;
	cudaMemcpy(h_mpo_2site.mpo2,h2,sizeof(float)*4,cudaMemcpyHostToDevice);

	float *iso[num_layers];
	int *sizeiso[num_layers];
	random_tree_tn_uniform(Ds,iso,sizeiso);
	for(unsigned i = 0; i < num_layers; ++i) {
		for(unsigned j = 0; j < 3; ++j) {
			cout<<sizeiso[i][j]<<"  ";
		}
		cout<<endl;
	}
	
	float *d_h1;
	cudaMalloc((void**)&d_h1,sizeof(float)*4);
	cudaMemcpy(d_h1,h1,sizeof(float)*4,cudaMemcpyHostToDevice);

	//printTensor(h_mpo_2site.mpo2,2,2,1);

	cout<<"************************************"<<endl;

	clock_t t1,t2;
 	double times=0.0;
 	t1=clock();
	opt_tree_energy(iso,d_h1,h_mpo_2site,num_sweeps,1,verbose,0.2,sizeiso);
	t2=clock();
    times = (double)(t2-t1)/CLOCKS_PER_SEC;
    cout<<"cost time is :"<<times<<endl;

	
}