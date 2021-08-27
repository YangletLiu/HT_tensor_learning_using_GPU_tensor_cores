#include "head.h"

void _descend_energy_env(float *env,half *ios,int *sizeiso,int size_env,cublasHandle_t handle,cutensorHandle_t tensor_handle)
{
	// 这里的函数_descend_energy_env_L 与_descend_energy_env_R  用的都是同一个函数
	float alpha = 1.0,beta = 0.0;
	//[iso_012, env, backend.conj(iso_012)], [(2, 3, -1), (2, 1), (1, 3, -2)])
	float *env_new;
	cudaMalloc((void**)&env_new,sizeof(float)*sizeiso[0]*sizeiso[0]);//iso里 size0与size1是一样的，所以这里L R都能用
	float *temp;
	cudaMalloc((void**)&temp,sizeof(float)*sizeiso[0]*sizeiso[1]*size_env);

	float *iso_f;
	cudaMalloc((void**)&iso_f,sizeof(float)*sizeiso[0]*sizeiso[1]*sizeiso[2]);
	h2f<<<512,512>>>(ios,iso_f,sizeiso[0]*sizeiso[1]*sizeiso[2]);

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeiso[0]*sizeiso[1],size_env,size_env,&alpha,iso_f,sizeiso[0]*sizeiso[1],env,size_env,
	            &beta,temp,sizeiso[0]*sizeiso[1]);
	
	vector<int> modeA{'a','b','c'};
  	vector<int> modeB{'a','d','c'};
  	vector<int> modeC{'b','d'};
  	unordered_map<int, int64_t> extent;
  	extent['a'] = sizeiso[0];
  	extent['b'] = sizeiso[1];
  	extent['c'] = sizeiso[2];
  	extent['d'] = sizeiso[1]; 
  	ncon_1(temp,iso_f,env_new,modeA,modeB,modeC,extent,tensor_handle); //出错
	
  	//reshape(temp,sizeiso[0],sizeiso[1],size_env);
  	//reshape(ios,sizeiso[0],sizeiso[1],sizeiso[2]);
  	
  	//cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[0],sizeiso[0],sizeiso[1]*sizeiso[2],
  	            //&alpha,temp,sizeiso[0],ios,sizeiso[0],
  	            //&beta,env_new,sizeiso[0]);

  	cudaMalloc((void**)&env,sizeof(float)*sizeiso[0]*sizeiso[0]);
  	cublasScopy(handle,sizeiso[0]*sizeiso[0],env_new,1,env,1);
  	cudaFree(env_new);
  	cudaFree(temp);
  	cudaFree(iso_f);


}

void _mpo_with_state(half *iso_012,half *iso_021,int *sizeiso,h2_mpo h2,int size_h2,float *states,int *size_state,
                     cublasHandle_t handle,cutensorHandle_t tensor_handle,float *envL,float *envR)
{
	float alpha = 1.0,beta = 0.0;
	//(1, 3), (1, -1, 2), (4, 2), (3, 4, -2)  states, iso_021, h2.mpo1, iso_012
	half *states_h,*temp2_h,*temp1_h;
	cudaMalloc((void**)&states_h,sizeof(half)*size_state[0]*size_state[1]);
	cudaMalloc((void**)&temp2_h,sizeof(half)*sizeiso[0]*size_h2*size_state[1]);
	cudaMalloc((void**)&temp1_h,sizeof(half)*sizeiso[1]*size_h2*size_state[1]);
	f2h(states,states_h,size_state[0]*size_state[1]);

	float *temp;
	cudaMalloc((void**)&temp,sizeof(float)*sizeiso[1]*sizeiso[0]*size_state[1]);
	/*cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeiso[1]*sizeiso[0],size_state[1],sizeiso[2],&alpha,iso_021,sizeiso[1]*sizeiso[0],
	            states,size_state[0],&beta,temp,sizeiso[1]*sizeiso[0]);*/

	 cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                           sizeiso[1]*sizeiso[0],size_state[1],sizeiso[2],
                           &alpha,iso_021,CUDA_R_16F,sizeiso[1]*sizeiso[0],
                           states_h,CUDA_R_16F,size_state[0],
                           &beta,temp,CUDA_R_32F,sizeiso[1]*sizeiso[0],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	


	float *temp1;
	cudaMalloc((void**)&temp1,sizeof(float)*sizeiso[1]*size_h2*size_state[1]);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[1],size_h2,sizeiso[0],&alpha,temp,sizeiso[1],sizeiso[1]*sizeiso[0],
	                          h2.mpo1,size_h2,0,&beta,temp1,sizeiso[1],sizeiso[1]*size_h2,size_state[1]);
	
	 f2h(temp1,temp1_h,sizeiso[1]*size_h2*size_state[1]);

	/*cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[1],sizeiso[1],size_h2*size_state[1],&alpha,temp1,sizeiso[1],iso_021,sizeiso[1],
	            &beta,envL,sizeiso[1]);*/
	   cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           sizeiso[1],sizeiso[1],size_h2*size_state[1],
                           &alpha,temp1_h,CUDA_R_16F,sizeiso[1],
                           iso_021,CUDA_R_16F,sizeiso[1],
                           &beta,envL,CUDA_R_32F,sizeiso[1],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	//====================================================================================
	//(1, 3), (1, -1, 2), (4, 2), (3, 4, -2) states, iso_012, h2.mpo2, iso_021

	/*cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeiso[0]*sizeiso[1],size_state[1],sizeiso[2],&alpha,iso_012,sizeiso[0]*sizeiso[1],
	            states,size_state[0],&beta,temp,sizeiso[0]*sizeiso[1]);*/
  	
  	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                           sizeiso[0]*sizeiso[1],size_state[1],sizeiso[2],
                           &alpha,iso_012,CUDA_R_16F,sizeiso[0]*sizeiso[1],
                           states_h,CUDA_R_16F,size_state[0],
                           &beta,temp,CUDA_R_32F,sizeiso[0]*sizeiso[1],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);


	float *temp2;
	cudaMalloc((void**)&temp2,sizeof(float)*sizeiso[0]*size_h2*size_state[1]);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[0],size_h2,sizeiso[1],&alpha,temp,sizeiso[0],sizeiso[0]*sizeiso[1],
	                          h2.mpo2,size_h2,0,&beta,temp2,sizeiso[0],sizeiso[0]*size_h2,size_state[1]);
	//这里与上边同理，直接与iso_012相乘

     f2h(temp2,temp2_h,sizeiso[0]*size_h2*size_state[1]);

	/*cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[0],sizeiso[0],size_h2*size_state[1],&alpha,temp2,sizeiso[0],iso_012,sizeiso[0],
	            &beta,envR,sizeiso[0]);*/
	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                           sizeiso[0],sizeiso[0],size_h2*size_state[1],
                           &alpha,temp2_h,CUDA_R_16F,sizeiso[0],
                           iso_012,CUDA_R_16F,sizeiso[0],
                           &beta,envL,CUDA_R_32F,sizeiso[0],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);

	cudaFree(temp);
	cudaFree(temp1);
	cudaFree(temp2);
	cudaFree(temp2_h);
	cudaFree(temp1_h);
	cudaFree(states_h);

}
float _compute_env(int lvl,h2_mpo *ops,bool reflect,int i,half **isos_012,half **isos_021,int **sizeiso,float *states,int *size_state,
                 cublasHandle_t handle,cutensorHandle_t tensor_handle,
                 int size_h,int *size_ops,float *iso_h2R_012,float *iso_h2L_012,float *env)
{	
	float alpha=1.0,beta = 0.0;
	h2_mpo h2 = ops[lvl+1];
	if (reflect)
	{	
		// mpo1 mpo2 交换
		float *sweap;
		cudaMalloc((void**)&sweap,sizeof(float)*size_ops[lvl]*size_ops[lvl]);
		cublasScopy(handle,size_ops[lvl]*size_ops[lvl],h2.mpo1,1,sweap,1);
		cublasScopy(handle,size_ops[lvl]*size_ops[lvl],h2.mpo2,1,h2.mpo1,1);
		cublasScopy(handle,size_ops[lvl]*size_ops[lvl],sweap,1,h2.mpo2,1);
		cudaDeviceSynchronize();
		cudaFree(sweap);
	}
	float *envL,*envR;
	cudaMalloc((void**)&envL,sizeof(float)*sizeiso[i+1+lvl][1]*sizeiso[i+1+lvl][1]);
	cudaMalloc((void**)&envR,sizeof(float)*sizeiso[i+1+lvl][0]*sizeiso[i+1+lvl][0]);
	_mpo_with_state(isos_012[i+1+lvl],isos_021[i+1+lvl],sizeiso[i+1+lvl],h2,size_ops[lvl],states,size_state,handle,tensor_handle,envL,envR);

	int size_envR = sizeiso[i+1+lvl][1];
	int size_envL = sizeiso[i+1+lvl][0];
	for(int lvl2 = lvl-1; lvl>=0; lvl2--) {
		if(lvl2 <0)  //当 lvl2=0 出错
			break;
		else{
			//isos_012[i+1+lvl2]
			if(reflect)
			{
				//[iso_012, env, backend.conj(iso_012)], [(2, 3, -1), (2, 1), (1, 3, -2)])
				_descend_energy_env(envR,isos_021[i+1+lvl2],sizeiso[i+1+lvl2],size_envR,handle,tensor_handle);
				size_envR = sizeiso[i+1+lvl2][0];
				_descend_energy_env(envL,isos_012[i+1+lvl2],sizeiso[i+1+lvl2],size_envL,handle,tensor_handle);
				size_envL = sizeiso[i+1+lvl2][1];
			}
			else{
				_descend_energy_env(envR,isos_012[i+1+lvl2],sizeiso[i+1+lvl2],size_envR,handle,tensor_handle);
				size_envR = sizeiso[i+1+lvl2][0];
				_descend_energy_env(envL,isos_021[i+1+lvl2],sizeiso[i+1+lvl2],size_envL,handle,tensor_handle);
				size_envL = sizeiso[i+1+lvl2][1];
			}
		}	
	}
	float *iso_h2_L,*iso_h2_R;
	cudaMalloc((void**)&iso_h2_L,sizeof(float)*size_h*sizeiso[i][1]*sizeiso[i][2]);
	cudaMalloc((void**)&iso_h2_R,sizeof(float)*size_h*sizeiso[i][1]*sizeiso[i][2]);
	if(reflect)
	{
		//iso_h2_L, iso_h2_R = iso_h2R_012, iso_h2L_012
		cublasScopy(handle,size_h*sizeiso[i][1]*sizeiso[i][2],iso_h2R_012,1,iso_h2_L,1);
		cublasScopy(handle,size_h*sizeiso[i][1]*sizeiso[i][2],iso_h2L_012,1,iso_h2_R,1);
	}else //iso_h2_L, iso_h2_R = iso_h2L_012, iso_h2R_012
	{
		cublasScopy(handle,size_h*sizeiso[i][1]*sizeiso[i][2],iso_h2R_012,1,iso_h2_R,1);
		cublasScopy(handle,size_h*sizeiso[i][1]*sizeiso[i][2],iso_h2L_012,1,iso_h2_L,1);
	}
	//(1, -1), (1, -2, -3) envL  iso_h2_R  也就是说此时的envL大小是 sizeiso[i][2](方阵)
	float *envL2,*envR2;
	cudaMalloc((void**)&envL2,sizeof(float)*size_h*sizeiso[i][1]*sizeiso[i][2]);
	cudaMalloc((void**)&envR2,sizeof(float)*size_h*sizeiso[i][1]*sizeiso[i][2]);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,size_h*sizeiso[i][1],sizeiso[i][2],sizeiso[i][2],&alpha,iso_h2_R,size_h*sizeiso[i][1],
	            envL,sizeiso[i][2],&beta,envL2,size_h*sizeiso[i][1]);

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,size_h*sizeiso[i][1],sizeiso[i][2],sizeiso[i][2],&alpha,iso_h2_L,size_h*sizeiso[i][1],
	            envR,sizeiso[i][2],&beta,envR2,size_h*sizeiso[i][1]);

	float weight; weight = 1/pow(2, lvl+1);
	alpha = weight;beta = weight;
	float *env_temp;
	cudaMalloc((void**)&env_temp,sizeof(float)*size_h*sizeiso[i][1]*sizeiso[i][2]);
	cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,size_h*sizeiso[i][1],sizeiso[i][2],&alpha,envL2,size_h*sizeiso[i][1],
	            &beta,envR2,size_h*sizeiso[i][1],env_temp,size_h*sizeiso[i][1]);
	alpha = 1.0;
	cublasSaxpy(handle,size_h*sizeiso[i][1]*sizeiso[i][2],&alpha,env_temp,1,env,1);


	cudaFree(envL2);
	cudaFree(envR2);
	cudaFree(iso_h2_L);
	cudaFree(iso_h2_R);
	cudaFree(envL);
	cudaFree(envR);
	cudaFree(env_temp);

	return weight;
}


void opt_energy_env_1site(half *iso_012,half *iso_021,float *hl1,h2_mpo &h2,float *states,int *sizeiso,int *size_state,
                          cublasHandle_t handle,cutensorHandle_t tensor_handle,int size_h,float *env,int flag)
{
	float alpha = 1.0,beta = 0.0,beta1 = 1.0;
	float *terms_012,*terms_021,*terms_021T;
	cudaMalloc((void**)&terms_012,sizeof(float)*size_h*size_h*sizeiso[2]);
  	cudaMalloc((void**)&terms_021,sizeof(float)*sizeiso[1]*size_h*sizeiso[2]);
  	cudaMalloc((void**)&terms_021T,sizeof(float)*sizeiso[1]*size_h*sizeiso[2]);
	int size_h_new = size_h;
	_ascend_uniform_op_to_1site_partial(hl1,h2,iso_012,iso_021,sizeiso,handle,tensor_handle,terms_012,terms_021,size_h,size_h_new,flag);
	//terms_012 + terms_021的 前两维转置

	mode2<<<512,512>>>(terms_021,terms_021T,sizeiso[1],size_h,sizeiso[2]);
	float *terms;
	cudaMalloc((void**)&terms,sizeof(float)*size_h*size_h*sizeiso[2]);

	cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,size_h*size_h,sizeiso[2],&alpha,terms_021T,size_h*size_h,
	            &beta1,terms_012,sizeiso[1]*size_h,terms,size_h*size_h);
	//float *env;
	//cudaMalloc((void**)&env,sizeof(float)*size_h*size_h*size_state[1]);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,size_h*size_h,size_state[1],size_state[0],
	            &alpha,terms,size_h*size_h,states,size_state[0],&beta,env,size_h*size_h);
	cudaFree(terms_012);
	cudaFree(terms_021);
	cudaFree(terms_021T);

}

void opt_energy_env_2site(int i,half **isos_012,half **isos_021,h2_mpo &hl2,float **states,
                          int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,int size_h,float *env2)
{
	//isos_012/021 是从 i开始的，states 是从 i+2 开始的
	int levels_above = num_layers-i-1;
	float alpha = 1.0,beta = 0.0;
	h2_mpo ops[num_layers-i+1];
	int size_ops[num_layers-i];
	size_ops[0]= size_h;

	ops[0].init1(size_ops[0],size_ops[0]);
	ops[0].init2(size_ops[0],size_ops[0]);

	cublasScopy(handle,size_h*size_h,hl2.mpo1,1,ops[0].mpo1,1);
	cublasScopy(handle,size_h*size_h,hl2.mpo2,1,ops[0].mpo2,1);

	for(int k = i;k<num_layers;k++)
	{
		//这里需要改，因为对于函数ascend_op_2site_to_2site变了实现
		ops[k-i+1].init1(sizeiso[k][2],sizeiso[k][2]);
		ops[k-i+1].init2(sizeiso[k][2],sizeiso[k][2]);
		ascend_op_2site_to_2site(ops[k-i],isos_012[k],isos_021[k],sizeiso[k],handle,ops[k-i+1],size_ops[k-i]);
		if(k-i<num_layers-1)
			size_ops[k-i +1] = sizeiso[k][2];
		//cout<<"--->"<<endl;printTensor(ops[k-i].mpo1,4,4,1);
		//cout<<"--->"<<endl;printTensor(ops[k-i].mpo2,4,4,1);
	}
	
	cudaDeviceSynchronize();
	//isos_021[i] hl2.mpo1 ()
	float *iso_h2R_012;// batch = 1，也就是hl2.mpo1只有一个，后面可能会有两个的情况
	cudaMalloc((void**)&iso_h2R_012,sizeof(float)*size_h*sizeiso[i][1]*sizeiso[i][2]);

	half *mpo1,*mpo2;
	cudaMalloc((void**)&mpo1,sizeof(half)*size_h*size_h);
	cudaMalloc((void**)&mpo2,sizeof(half)*size_h*size_h);

	f2h(hl2.mpo2,mpo2,size_h*size_h);
	f2h(hl2.mpo1,mpo1,size_h*size_h);

	/*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h,sizeiso[i][1],size_h,&alpha,hl2.mpo2,size_h,0,
	                          isos_021[i],sizeiso[i][1],sizeiso[i][1]*sizeiso[i][0],&beta,iso_h2R_012,size_h,size_h*sizeiso[i][1],sizeiso[i][2]);*/
	
	  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_T,
		             size_h,sizeiso[i][1],size_h,
		             &alpha,mpo2,CUDA_R_16F,size_h,0,
		             isos_021[i],CUDA_R_16F,sizeiso[i][1],sizeiso[i][1]*sizeiso[i][0],
		             &beta,iso_h2R_012,CUDA_R_32F,size_h,size_h*sizeiso[i][1],sizeiso[i][2],		             
		             CUDA_R_32F,
		             CUBLAS_GEMM_DEFAULT_TENSOR_OP);


	float *iso_h2L_012;
	cudaMalloc((void**)&iso_h2L_012,sizeof(float)*sizeiso[i][0]*size_h*sizeiso[i][2]);


	/*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[i][0],size_h,sizeiso[i][1],
	                          &alpha,isos_012[i],sizeiso[i][0],sizeiso[i][0]*sizeiso[i][1],hl2.mpo1,size_h,0,
	                          &beta,iso_h2L_012,sizeiso[i][0],sizeiso[i][0]*size_h,sizeiso[0][2]);*/

	  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_T,
		             sizeiso[i][0],size_h,sizeiso[i][1],
		             &alpha,isos_012[i],CUDA_R_16F,sizeiso[i][0],sizeiso[i][0]*sizeiso[i][1],
		             mpo1,CUDA_R_16F,size_h,0,
		             &beta,iso_h2L_012,CUDA_R_32F,sizeiso[i][0],sizeiso[i][0]*size_h,sizeiso[0][2],		             
		             CUDA_R_32F,
		             CUBLAS_GEMM_DEFAULT_TENSOR_OP);                         

	float *env;
	float weightsum=0,weight=0;
	cudaMalloc((void**)&env,sizeof(float)*size_h*sizeiso[i][1]*sizeiso[i][2]);


	for(unsigned lvl = 0; lvl < levels_above; ++lvl) {
		weight=_compute_env(lvl,ops,false,i,isos_012,isos_021,sizeiso,states[i+2+lvl],size_state[i+2+lvl],
		             handle,tensor_handle,size_h,size_ops,iso_h2R_012,iso_h2L_012,env);
		weightsum = weightsum+weight;
	}
	int lvl = levels_above-1;
	weight=_compute_env(lvl,ops,true,i,isos_012,isos_021,sizeiso,states[i+2+lvl],size_state[i+2+lvl],
		             handle,tensor_handle,size_h,size_ops,iso_h2R_012,iso_h2L_012,env);


	cublasScopy(handle,size_h*sizeiso[i][1]*sizeiso[i][2],env,1,env2,1);

	cudaFree(iso_h2R_012);
	cudaFree(iso_h2L_012);
	cudaFree(env);

}


void opt_energy_env(int i,half **isos_012,half **isos_021,float *hl1,h2_mpo &hl2,float **states,
                           int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,int size_h,float *env)
{
	float alpha = 1.0,beta1 = 1.0;
	if(i == num_layers-1) // i 为 numlayer-1，最后一次循环
	{
		
		opt_energy_env_1site(isos_012[i],isos_021[i],hl1,hl2,states[i+1],sizeiso[i],size_state[i+1],handle,tensor_handle,size_h,env,1);
		
	}else
	{
		float *env1,*env2;
		cudaMalloc((void**)&env1,sizeof(float)*size_h*size_h*size_state[i+1][1]);
		cudaMalloc((void**)&env2,sizeof(float)*size_h*size_h*size_state[i+1][1]);

		opt_energy_env_1site(isos_012[i],isos_021[i],hl1,hl2,states[i+1],sizeiso[i],size_state[i+1],handle,tensor_handle,size_h,env1,0);

		//isos_012/021 是从 i开始的，states 是从 i+2 开始的
		//主要问题在这里
		opt_energy_env_2site(i,isos_012,isos_021,hl2,states,sizeiso,size_state,handle,tensor_handle,size_h,env2);// 问题！
		
		cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,size_h*size_h,size_state[i+1][1],&alpha,env1,size_h*size_h,
		            &beta1,env2,size_h*size_h,env,size_h*size_h);		
	}

}


void opt_energy_layer_once(int i,half **isos_012,half **isos_021,float *hl1,h2_mpo &hl2,float **states,
                           int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,cusolverDnHandle_t cusolverH,int size_h,
                           float *iso_new,float *svs)
{
	// opt_energy_env 函数
	float alpha = 1.0,beta = 0.0;
	float *env,*envT;
	cudaMalloc((void**)&env,sizeof(float)*size_h*size_h*size_state[i+1][1]);
	cudaMalloc((void**)&envT,sizeof(float)*size_h*size_h*size_state[i+1][1]);
	opt_energy_env(i,isos_012,isos_021,hl1,hl2,states,sizeiso,size_state,handle,tensor_handle,size_h,env);

	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,size_state[i+1][1],size_h*size_h,&alpha,env,size_h*size_h,&beta,env,size_h*size_h,
	            envT,size_state[i+1][1]);
	// env 的 size  size_h*size_h*size_state[i+1][1] 
	//需要的 env 的size 为 size_state[i+1][1] * （size_h*size_h） 但此时 m<n，我们对其转置后的做svd
	float *d_U,*S,*V;
	cudaMalloc ((void**)&d_U,sizeof(float)*size_state[i+1][1]*size_state[i+1][1]);
    cudaMalloc ((void**)&S,sizeof(float)*size_state[i+1][1]);
    cudaMalloc ((void**)&V,sizeof(float)*size_h*size_h*size_state[i+1][1]);

	gesvdj(envT,d_U,V,S,size_state[i+1][1],size_h*size_h, cusolverH);

	cudaDeviceSynchronize();
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h*size_h,size_state[i+1][1],size_state[i+1][1],
	            &alpha,V,size_h*size_h,d_U,size_state[i+1][1],&beta,iso_new,size_h*size_h);
	//printTensor(iso_new,3,3,1);
	cublasScopy(handle,size_state[i+1][1],S,1,svs,1);

	cudaFree(d_U);
	cudaFree(S);
	cudaFree(V);
	cudaFree(env);
	cudaFree(envT);

}

void opt_energy_layer(int i,half **isos_012,half **isos_021,float **states,float *hl1,h2_mpo &hl2,
                      int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,cusolverDnHandle_t cusolverH,int size_h,
                      float *iso_new,float *svs)
{
	// 需要的 isos_012是 i~num_layers 这里去第 i个为 iso_012

	opt_energy_layer_once(i,isos_012,isos_021,hl1,hl2,states,sizeiso,size_state,handle,tensor_handle,cusolverH,size_h,iso_new,svs);
	
}