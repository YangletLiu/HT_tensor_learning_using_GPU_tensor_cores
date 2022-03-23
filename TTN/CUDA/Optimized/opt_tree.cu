#include "head.h"


void _ascend_op_2site_to_1site_partial(h2_mpo &h2,half *iso_021,int *sizeiso,
                                         cublasHandle_t handle,cutensorHandle_t tensor_handle,
                                         int size_h,int size_h_new,float *iso_op_2site_012,int flag)
{
  /***
  * 返回  size_h * size_h *sizeiso[2]
  ****/
  float alpha = 1.0,beta = 0.0;
  if(flag == 0){
    float *iso_op_mpo_L_012;// 2 * size[1] * size[2]
    cudaMalloc((void**)&iso_op_mpo_L_012,sizeof(float)*sizeiso[1]*size_h*sizeiso[2]);   
    //（-1，-3，1）（-2，1） python ==> (-2,1,-3) (-1,1) CUDA
    /*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h,sizeiso[1],size_h,
                              &alpha,h2.mpo1,size_h,0,iso_021,sizeiso[1],sizeiso[1]*sizeiso[0],
                              &beta,iso_op_mpo_L_012,size_h,size_h*sizeiso[1],sizeiso[2]);*/
    half *mpo1;
    cudaMalloc((void**)&mpo1,sizeof(half)*size_h*size_h);
    f2h(h2.mpo1,mpo1,size_h*size_h);
    cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                 size_h,sizeiso[1],size_h,
                 &alpha,mpo1,CUDA_R_16F,size_h,0,
                 iso_021,CUDA_R_16F,sizeiso[1],sizeiso[1]*sizeiso[0],
                 &beta,iso_op_mpo_L_012,CUDA_R_32F,size_h,size_h*sizeiso[1],sizeiso[2],                 
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    //_ascend_partial(op,iso)   h[1] iso_op_mpo_L_012
    //float *iso_op_2site_012;
    //cudaMalloc((void**)&iso_op_2site_012,sizeof(float)*size_h*size_h*sizeiso[2]);
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h,size_h,sizeiso[1],&alpha,iso_op_mpo_L_012,size_h,size_h*sizeiso[1],
                              h2.mpo2,size_h,0,&beta,iso_op_2site_012,size_h,size_h*size_h,sizeiso[2]);
    cudaFree(iso_op_mpo_L_012);
    cudaFree(mpo1);
  }else
  {
    float *iso_op_mpo_L_012,*temp_result;// 2 * size[1] * size[2]
    cudaMalloc((void**)&iso_op_mpo_L_012,sizeof(float)*sizeiso[1]*size_h*sizeiso[2]);
    cudaMalloc((void**)&temp_result,sizeof(float)*size_h*size_h*sizeiso[2]);
    half *mpo1,*mpo2;
    cudaMalloc((void**)&mpo1,sizeof(half)*size_h*size_h);
    cudaMalloc((void**)&mpo2,sizeof(half)*size_h*size_h);
    f2h(h2.mpo1,mpo1,size_h*size_h);
    f2h(h2.mpo2,mpo2,size_h*size_h);
    /*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h,sizeiso[1],size_h,
                              &alpha,h2.mpo1,size_h,0,iso_021,sizeiso[1],sizeiso[1]*sizeiso[0],
                              &beta,iso_op_mpo_L_012,size_h,size_h*sizeiso[1],sizeiso[2]);*/
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_T,
                 size_h,sizeiso[1],size_h,
                 &alpha,mpo1,CUDA_R_16F,size_h,0,
                 iso_021,CUDA_R_16F,sizeiso[1],sizeiso[1]*sizeiso[0],
                 &beta,iso_op_mpo_L_012,CUDA_R_32F,size_h,size_h*sizeiso[1],sizeiso[2],                 
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h,size_h,sizeiso[1],&alpha,iso_op_mpo_L_012,size_h,size_h*sizeiso[1],
                              h2.mpo2,size_h,0,&beta,temp_result,size_h,size_h*size_h,sizeiso[2]);

    /*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h,sizeiso[1],size_h,
                              &alpha,h2.mpo2,size_h,0,iso_021,sizeiso[1],sizeiso[1]*sizeiso[0],
                              &beta,iso_op_mpo_L_012,size_h,size_h*sizeiso[1],sizeiso[2]);*/
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_T,
                 size_h,sizeiso[1],size_h,
                 &alpha,mpo2,CUDA_R_16F,size_h,0,
                 iso_021,CUDA_R_16F,sizeiso[1],sizeiso[1]*sizeiso[0],
                 &beta,iso_op_mpo_L_012,CUDA_R_32F,size_h,size_h*sizeiso[1],sizeiso[2],                 
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,size_h,size_h,sizeiso[1],&alpha,iso_op_mpo_L_012,size_h,size_h*sizeiso[1],
                              h2.mpo1,size_h,0,&beta,iso_op_2site_012,size_h,size_h*size_h,sizeiso[2]);
    cublasSaxpy(handle,size_h*size_h*sizeiso[2],&alpha,temp_result,1,iso_op_2site_012,1);

    cudaFree(iso_op_mpo_L_012);
    cudaFree(temp_result);
    cudaFree(mpo1);cudaFree(mpo2);

  }

}

void _ascend_uniform_op_to_1site_partial(float *h1,h2_mpo h_mpo_2site,half *iso_012,half *iso_021,int *sizeiso,cublasHandle_t handle,
                                         cutensorHandle_t tensor_handle,float *terms_012,float *iso_op_1site_L_021,int size_h,int size_h_new,int flag)
{ 
  float alpha = 1.0,beta = 0.0;
  float *iso_op_2site_012;
  cudaMalloc((void**)&iso_op_2site_012,sizeof(float)*size_h*size_h*sizeiso[2]);
  _ascend_op_2site_to_1site_partial(h_mpo_2site,iso_021,sizeiso,handle,tensor_handle,size_h,size_h_new,iso_op_2site_012,flag);

  float *iso_op_1site_R_012;
  cudaMalloc((void**)&iso_op_1site_R_012,sizeof(float)*sizeiso[0]*size_h*sizeiso[2]);
  half *h_h1;
  cudaMalloc((void**)&h_h1,sizeof(half)*size_h*size_h);
  f2h(h1,h_h1,size_h*size_h);
  /*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[0],size_h,sizeiso[1],&alpha,iso_012,sizeiso[0],sizeiso[0]*sizeiso[1],
                            h1,size_h,0,&beta,iso_op_1site_R_012,sizeiso[0],sizeiso[0]*size_h,sizeiso[2]);*/
   cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_T,
                 sizeiso[0],size_h,sizeiso[1],
                 &alpha,iso_012,CUDA_R_16F,sizeiso[0],sizeiso[0]*sizeiso[1],
                 h_h1,CUDA_R_16F,size_h,0,
                 &beta,iso_op_1site_R_012,CUDA_R_32F,sizeiso[0],sizeiso[0]*size_h,sizeiso[2],                 
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);                         
  /*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,sizeiso[1],size_h,sizeiso[0],&alpha,iso_021,sizeiso[1],sizeiso[1]*sizeiso[0],
                            h1,size_h,0,&beta,iso_op_1site_L_021,sizeiso[1],sizeiso[1]*size_h,sizeiso[2]);*/
   cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_T,
                 sizeiso[1],size_h,sizeiso[0],
                 &alpha,iso_021,CUDA_R_16F,sizeiso[1],sizeiso[1]*sizeiso[0],
                 h_h1,CUDA_R_16F,size_h,0,
                 &beta,iso_op_1site_L_021,CUDA_R_32F,sizeiso[1],sizeiso[1]*size_h,sizeiso[2],                 
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  //iso_op_2site_012 + iso_op_1site_R_012
  //float *terms_012;
  float beta1 = 1.0;
  
  cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,size_h,size_h*sizeiso[2],&alpha,iso_op_2site_012,size_h,
              &beta1,iso_op_1site_R_012,sizeiso[0],terms_012,size_h);
  cudaDeviceSynchronize();

}


void ascend_uniform_op_to_1site(float *h1,h2_mpo h_mpo_2site,half *iso_012,half *iso_021,int *sizeiso,
                                  cublasHandle_t handle,cutensorHandle_t tensor_handle, float *h1_new,int size_h,int size_h_new,int flag)
{
  float alpha = 1.0,beta = 0.0,beta1 = 1.0;
  float *terms_012,*iso_op_1site_L_021;
  cudaMalloc((void**)&terms_012,sizeof(float)*size_h*size_h*sizeiso[2]);
  cudaMalloc((void**)&iso_op_1site_L_021,sizeof(float)*sizeiso[1]*size_h*sizeiso[2]);
  half *iso_op_1site_L_0213,*terms_0123;
  cudaMalloc((void**)&iso_op_1site_L_0213,sizeof(half)*sizeiso[1]*size_h*sizeiso[2]);
  cudaMalloc((void**)&terms_0123,sizeof(half)*size_h*size_h*sizeiso[2]);
  _ascend_uniform_op_to_1site_partial(h1,h_mpo_2site,iso_012,iso_021,sizeiso,handle,tensor_handle,terms_012,iso_op_1site_L_021,size_h,size_h_new,flag);
  f2h(iso_op_1site_L_021,iso_op_1site_L_0213,sizeiso[1]*size_h*sizeiso[2]);
  f2h(terms_012,terms_0123,size_h*size_h*sizeiso[2]);
  //_complete_partial_ascend(iso_op_1site_L_021, iso_021)  iso * iso_op  反着的
  //float *res;
  //cudaMalloc((void**)&res,sizeof(float)*sizeiso[2]*sizeiso[2]);
  /*cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,sizeiso[2],sizeiso[2],sizeiso[1]*sizeiso[0],
              &alpha,iso_021,sizeiso[1]*sizeiso[0],iso_op_1site_L_021,sizeiso[1]*size_h,&beta,h1_new,sizeiso[2]);*/

    cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           sizeiso[2],sizeiso[2],sizeiso[1]*sizeiso[0],
                           &alpha,iso_021,CUDA_R_16F,sizeiso[1]*sizeiso[0],
                           iso_op_1site_L_0213,CUDA_R_16F,sizeiso[1]*size_h,
                           &beta,h1_new,CUDA_R_32F,sizeiso[2],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  //res += _complete_partial_ascend(terms_012, iso_012)
  /*cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,sizeiso[2],sizeiso[2],size_h*size_h,
              &alpha,iso_012,sizeiso[0]*sizeiso[1],terms_012,size_h*size_h,&beta1,h1_new,sizeiso[2]);*/
  cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           sizeiso[2],sizeiso[2],size_h*size_h,
                           &alpha,iso_012,CUDA_R_16F,sizeiso[0]*sizeiso[1],
                           terms_0123,CUDA_R_16F,size_h*size_h,
                           &beta1,h1_new,CUDA_R_32F,sizeiso[2],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);          

  cudaDeviceSynchronize();
  cudaFree(terms_012);cudaFree(terms_0123);
  cudaFree(iso_op_1site_L_021);cudaFree(iso_op_1site_L_0213);

  //return res;
}

void ascend_op_2site_to_2site(h2_mpo h_mpo_2site,half *iso_012,half *iso_021,int *sizeiso,
                              cublasHandle_t handle,h2_mpo &h_mpo_2site_new,int size_h)
{
  // iso_021 , h2[1] , iso_021   (-1, 3, 1), (1, 2),(-2, 3, 2)
  float alpha = 1.0,beta = 0.0;
  float *tmp;
  cudaMalloc((void**)&tmp,sizeof(float)*sizeiso[1]*size_h*sizeiso[2]); // 此时的size[0] 与 size[1] 相等

  half *mpo2,*mpo1,*tmp3;
  cudaMalloc((void**)&mpo1,sizeof(half)*size_h*size_h);
  cudaMalloc((void**)&mpo2,sizeof(half)*size_h*size_h);
  cudaMalloc((void**)&tmp3,sizeof(half)*sizeiso[1]*size_h*sizeiso[2]);
  f2h(h_mpo_2site.mpo1,mpo1,size_h*size_h);
  f2h(h_mpo_2site.mpo2,mpo2,size_h*size_h);
 // h_mpo_2site_new.init1(sizeiso[2],sizeiso[2]);
 // h_mpo_2site_new.init2(sizeiso[2],sizeiso[2]);
  
  /*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeiso[1],size_h,size_h,&alpha,iso_021,sizeiso[1],sizeiso[1]*sizeiso[0],
                            h_mpo_2site.mpo2,size_h,0,&beta,tmp,sizeiso[1],sizeiso[1]*size_h,sizeiso[2]);*/
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_N,
                 sizeiso[1],size_h,size_h,
                 &alpha,iso_021,CUDA_R_16F,sizeiso[1],sizeiso[1]*sizeiso[0],
                 mpo2,CUDA_R_16F,size_h,0,
                 &beta,tmp,CUDA_R_32F,sizeiso[1],sizeiso[1]*size_h,sizeiso[2],                 
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    f2h(tmp,tmp3,sizeiso[1]*size_h*sizeiso[2]);

  /*cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,sizeiso[2],sizeiso[2],sizeiso[0]*sizeiso[1],&alpha,tmp,size_h*sizeiso[1],
              iso_021,sizeiso[1]*sizeiso[0],&beta,h_mpo_2site_new.mpo2,sizeiso[2]);*/
      cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           sizeiso[2],sizeiso[2],sizeiso[0]*sizeiso[1],
                           &alpha,tmp3,CUDA_R_16F,size_h*sizeiso[1],
                           iso_021,CUDA_R_16F,sizeiso[1]*sizeiso[0],
                           &beta,h_mpo_2site_new.mpo2,CUDA_R_32F,sizeiso[2],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);        

  cudaDeviceSynchronize();

  /*cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeiso[0],size_h,size_h,&alpha,iso_012,sizeiso[0],sizeiso[0]*sizeiso[1],
                            h_mpo_2site.mpo1,size_h,0,&beta,tmp,sizeiso[0],sizeiso[0]*size_h,sizeiso[2]);*/
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N,CUBLAS_OP_N,
                 sizeiso[0],size_h,size_h,
                 &alpha,iso_012,CUDA_R_16F,sizeiso[0],sizeiso[0]*sizeiso[1],
                 mpo1,CUDA_R_16F,size_h,0,
                 &beta,tmp,CUDA_R_32F,sizeiso[0],sizeiso[0]*size_h,sizeiso[2],                 
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);  
       f2h(tmp,tmp3,sizeiso[1]*size_h*sizeiso[2]);                                
  /*cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,sizeiso[2],sizeiso[2],sizeiso[0]*sizeiso[1],&alpha,tmp,sizeiso[0]*sizeiso[1],
              iso_012,sizeiso[1]*sizeiso[0],&beta,h_mpo_2site_new.mpo1,sizeiso[2]);*/
   
   cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                           sizeiso[2],sizeiso[2],sizeiso[0]*sizeiso[1],
                           &alpha,tmp3,CUDA_R_16F,sizeiso[0]*sizeiso[1],
                           iso_012,CUDA_R_16F,sizeiso[1]*sizeiso[0],
                           &beta,h_mpo_2site_new.mpo1,CUDA_R_32F,sizeiso[2],
                           CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);

   cudaDeviceSynchronize();
   cudaFree(tmp);cudaFree(tmp3);
   cudaFree(mpo1);cudaFree(mpo2);
   //return h_mpo_2site_new;

}

void ascend_uniform_op_local(float *h1,h2_mpo h_mpo_2site,half *iso_012,half *iso_021,int *sizeiso,cublasHandle_t handle,cutensorHandle_t tensor_handle,
                             float *h1_new,h2_mpo &h_mpo_2site_new,int size_h,int size_h_new)
{
  //函数中只有 h1 变化  2*2 变 4*4  
  ascend_uniform_op_to_1site(h1,h_mpo_2site,iso_012,iso_021,sizeiso, handle, tensor_handle,h1_new,size_h,size_h_new,0);

  //函数中只有 h_mpo_2site变化
  ascend_op_2site_to_2site(h_mpo_2site,iso_012,iso_021,sizeiso,handle,h_mpo_2site_new,size_h);

}

void descend_state_1site(float *states_in,float *iso_012,float *iso_021,int *sizeiso,int *sizein,float *states_out,int *sizeout,cublasHandle_t handle,cutensorHandle_t tensor_handle)
{
  //descend_state_1site_L states_in   iso_021 -> state_1L
  float alpha = 1.0,beta = 0.0;
  float *state_1L,*state_1R;
  cudaMalloc((void**)&state_1L,sizeof(float)*sizeiso[0]*sizeiso[0]);
  cudaMalloc((void**)&state_1R,sizeof(float)*sizeiso[1]*sizeiso[1]);
  // iso state iso (2, 3, -1), (2, 1), (1, 3, -2)
  float *temp;
  cudaMalloc((void**)&temp,sizeof(float)*sizeiso[1]*sizeiso[0]*sizein[1]);
  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeiso[1]*sizeiso[0],sizein[1],sizeiso[2],
              &alpha,iso_021,sizeiso[1]*sizeiso[0],states_in,sizein[0],&beta,temp,sizeiso[1]*sizeiso[0]);
  // CUDA的顺序： (3,-1,1) (3,-2,1) temp*iso_021
  // iso_021需要mode2 转置，也就是 iso_012

  vector<int> modeA{'a','b','c'};
  vector<int> modeB{'a','d','c'};
  vector<int> modeC{'b','d'};
  unordered_map<int, int64_t> extent;
  extent['a'] = sizeiso[1];
  extent['b'] = sizeiso[0];
  extent['c'] = sizein[1];
  extent['d'] = sizeiso[0];
  ncon_1(temp,iso_021,state_1L,modeA,modeB,modeC,extent,tensor_handle);
  //
  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeiso[0]*sizeiso[1],sizein[1],sizeiso[2],
              &alpha,iso_012,sizeiso[0]*sizeiso[1],states_in,sizein[0],&beta,temp,sizeiso[0]*sizeiso[1]);

  vector<int> modeA2={'a','b','c'};
  vector<int> modeB2={'a','d','c'};
  vector<int> modeC2={'b','d'};

  extent['a'] = sizeiso[0];
  extent['b'] = sizeiso[1];
  extent['c'] = sizein[1];
  extent['d'] = sizeiso[1];

  ncon_1(temp,iso_012,state_1R,modeA2,modeB2,modeC2,extent,tensor_handle);
  float alpha1 = 0.5,beta1 = 0.5;
  cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,sizeout[0],sizeout[1],&alpha1,state_1L,sizeout[0],&beta1,state_1L,sizeout[0],
              states_out,sizeout[0]);

  cudaFree(temp);
  cudaFree(state_1L);
  cudaFree(state_1R);
}


void opt_tree_energy(float **iso_012,float *h1,h2_mpo h_mpo_2site,int num_sweeps,int itr_l,int verbose,float ham_shift,int **sizeiso)
{
	//shift_ham  ham_shift ！=‘auto’ h1 对角线元素减去ham_shift
  //两个H一开始大小均为 2*2   不会变
	shift_ham<<<512,512>>>(h1,2,ham_shift);
  cudaDeviceSynchronize();
	int L = num_layers; 
	int bottom = 0;
  float alpha = 1.0,beta = 0.0;
  float *en=NULL;
  dim3 block0((sizeiso[num_layers-2][0]*sizeiso[num_layers-2][1]*sizeiso[num_layers-2][2]+1024-1)/1024,1,1); 
  half *isoh_012[num_layers];
  half *isoh_021[num_layers];
  for(unsigned i = 0; i < num_layers; ++i) {
    cudaMalloc((void**)&isoh_012[i],sizeof(half)*sizeiso[i][0]*sizeiso[i][1]*sizeiso[i][2]);
    cudaMalloc((void**)&isoh_021[i],sizeof(half)*sizeiso[i][0]*sizeiso[i][1]*sizeiso[i][2]);
    f2h(iso_012[i],isoh_012[i],sizeiso[i][0]*sizeiso[i][1]*sizeiso[i][2]);
    cudaDeviceSynchronize();
    mode2h<<<block0,1024>>>(isoh_012[i],isoh_021[i],sizeiso[i][0],sizeiso[i][1],sizeiso[i][2]);
    cudaDeviceSynchronize();
  }

  // transpose(isos_012[l], (0, 2, 1))
  // iso_021[i] 的 size是 sizeiso[i][1] sizeiso[i][0] sizeiso[i][2]
  float *iso_021[num_layers];

  for(unsigned i = 0; i < num_layers; ++i) {
      cudaMalloc((void**)&iso_021[i],sizeof(float)*sizeiso[i][0]*sizeiso[i][1]*sizeiso[i][2]);
      mode2<<<512,512>>>(iso_012[i],iso_021[i],sizeiso[i][0],sizeiso[i][1],sizeiso[i][2]);
      cudaDeviceSynchronize();
    }
  cublasHandle_t handle;
  cublasCreate(&handle);

  cutensorHandle_t tensor_handle;
  cutensorInit(&tensor_handle);
  cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);


  float *h1_new;
  h2_mpo h_mpo_2site_new;
  int size_h = 2;
  int size_h_new ;

	for(int l=0;l<L;l++)  //需要 iso_012每一个tensor 的size
	{
    if(sizeiso[l][2] == sizeiso[l][0] * sizeiso[l][1])
    { 
      //  这个函数中 只有 h1，h_mpo_2site 会变化，iso等tensor均不变
      cudaMalloc((void**)&h1_new,sizeof(float)*sizeiso[l][2]*sizeiso[l][2]);
      size_h_new = sizeiso[l][2];
      h_mpo_2site_new.init1(sizeiso[l][2],sizeiso[l][2]);
      h_mpo_2site_new.init2(sizeiso[l][2],sizeiso[l][2]);
      
      ascend_uniform_op_local(h1,h_mpo_2site,isoh_012[l],isoh_021[l],sizeiso[l],handle,tensor_handle,h1_new,h_mpo_2site_new,size_h,size_h_new);
      bottom = l+1;
    }
    else break;
	}
  cudaFree(h1);cudaFree(h_mpo_2site.mpo1);cudaFree(h_mpo_2site.mpo2);
//===================================================================================
  float hh[16] ={-2.4,0,0,-1,0, -0.40000004,-1,0,0,-1, -0.40000004,0,-1,0,0,1.6};
  //hh={-2.4,0,0,-1,0, -0.40000004,-1,0,0,-1, -0.40000004,0,-1,0,0,1.6};
  cudaMemcpy(h1_new,hh,sizeof(float)*16,cudaMemcpyHostToDevice);
  float hh2[16] = {0,-1,0,0,-1,0,0,0,0,0,0,-1,0,0,-1,0};
  cudaMemcpy(h_mpo_2site_new.mpo1,hh2,sizeof(float)*16,cudaMemcpyHostToDevice);
  float hh3[16] = {0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0};
  cudaMemcpy(h_mpo_2site_new.mpo2,hh3,sizeof(float)*16,cudaMemcpyHostToDevice);
  

//====================================================================================

  //下面开始 sweep 的过程
  
  float *states[num_layers+1];  // num_layers = 6,iso里有6个，states里有7个
  int *size_state[num_layers+1];
  float *hl1;
  cudaMalloc((void**)&hl1,sizeof(float)*16);
  h2_mpo hl2;
  

  float *svs[L];
  float *iso_new;
  float *h1_up;
  h2_mpo h2_up;
  float *min;
  cudaMalloc((void**)&min,sizeof(float)*(L-1));
  int min_ind;
  float *min_sv = new float[1];
  //clock_t t1,t2;
  //double times=0.0;
cout<<"----------------------sweep----------------------"<<endl;
  for(int  sw = 0; sw < num_sweeps; ++sw) {  // sweep
    hl2.init1(4,4);
    hl2.init2(4,4);
    //t1=clock();
    //states = all_states_1site(isos_012[bottom:])
    cudaHostAlloc((void**)&size_state[num_layers],sizeof(int)*2,0);
    cudaMalloc((void**)&states[num_layers],sizeof(float)*sizeiso[num_layers-1][2]*sizeiso[num_layers-1][2]);
    
    //dim3 block0((sizeiso[num_layers-1][2]*sizeiso[num_layers-1][2]+1024-1)/1024,1,1);
    initIdeMat<<<512,512>>>(states[num_layers],sizeiso[num_layers-1][2]);
    cout<<"ko ko de u~~~~~~"<<endl;
    size_state[num_layers][0] = 1;
    size_state[num_layers][1] = 1;
    cout<<"ko ko de u~~~~~~111"<<endl;
    //iso的后5个
    //cout<<sizeiso[num_layers-1][2]<<endl;

    for(int j=num_layers-1;j>0;j--)
    {
      //states[0] 目前为 None
      //输入为states[j+1] iso_012[j] iso_021[j],输出 states[j]
      
      cudaHostAlloc((void**)&size_state[j],sizeof(int)*2,0);

      size_state[j][0] = sizeiso[j][1];
      size_state[j][1] = sizeiso[j][1];
     
      cudaMalloc((void**)&states[j],sizeof(float)*size_state[j][0]*size_state[j][1]);

      descend_state_1site(states[j+1],iso_012[j],iso_021[j],sizeiso[j],size_state[j+1],states[j],size_state[j],handle,tensor_handle);
      
    }
    //opt_energy_layer  核心

    cublasScopy(handle,16,h1_new,1,hl1,1);
    cublasScopy(handle,16,h_mpo_2site_new.mpo1,1,hl2.mpo1,1);
    cublasScopy(handle,16,h_mpo_2site_new.mpo2,1,hl2.mpo2,1);
    size_h = 4;

    for(unsigned i = bottom; i < L; ++i) {

      //cout<<"============Optimizing level  "<<i<<" ===========size_h is :"<<size_h<<endl;
      cudaMalloc((void**)&iso_new,sizeof(float)*size_h*size_h*size_state[i+1][1]);
      cudaMalloc((void**)&svs[i],sizeof(float)*size_state[i+1][1]);
      opt_energy_layer(i,isoh_012,isoh_021,states,hl1,hl2,sizeiso,size_state,handle,tensor_handle,cusolverH,size_h,iso_new,svs[i]);
      //这里之后  iso_012 i+1个已经为0
      //cout<<"=======over one step======="<<endl;

      cublasIsamin(handle,size_state[i+1][1],svs[i],1,&min_ind);
      cublasScopy(handle,1,svs[i]+min_ind,1,min+i-1,1);

      cublasScopy(handle,size_h*size_h*size_state[i+1][1],iso_new,1,iso_012[i],1);
      mode2<<<512,512>>>(iso_new,iso_021[i],size_h,size_h,size_state[i+1][1]);

      f2h(iso_012[i],isoh_012[i],sizeiso[i][0]*sizeiso[i][1]*sizeiso[i][2]);
      f2h(iso_021[i],isoh_021[i],sizeiso[i][0]*sizeiso[i][1]*sizeiso[i][2]);
      //printTensor(iso_new,4,4,1);
      if(i<L-1)
      {
        cudaMalloc((void**)&h1_up,sizeof(float)*sizeiso[i][2]*sizeiso[i][2]);

        h2_up.init1(sizeiso[i][2],sizeiso[i][2]);
        h2_up.init2(sizeiso[i][2],sizeiso[i][2]);
        size_h_new = sizeiso[i][2];
        ascend_uniform_op_local(hl1,hl2,isoh_012[i],isoh_021[i],sizeiso[i],handle,tensor_handle,h1_up,h2_up,size_h,size_h_new);
        cudaFree(hl1);cudaFree(hl2.mpo1);cudaFree(hl2.mpo2);

        cudaMalloc((void**)&hl1,sizeof(float)*sizeiso[i][2]*sizeiso[i][2]);
        hl2.init1(sizeiso[i][2],sizeiso[i][2]);
        hl2.init2(sizeiso[i][2],sizeiso[i][2]);
        cublasScopy(handle,sizeiso[i][2]*sizeiso[i][2],h1_up,1,hl1,1);
        cublasScopy(handle,sizeiso[i][2]*sizeiso[i][2],h2_up.mpo1,1,hl2.mpo1,1);
        cublasScopy(handle,sizeiso[i][2]*sizeiso[i][2],h2_up.mpo2,1,hl2.mpo2,1);
        size_h = size_h_new;
        cudaFree(h1_up);cudaFree(h2_up.mpo1);cudaFree(h2_up.mpo2);
      }
      
    }
    //借用 h2_up
     
    size_h_new = sizeiso[num_layers-1][2];
    cudaMalloc((void**)&h1_up,sizeof(float)*size_h_new*size_h_new);

    ascend_uniform_op_to_1site(hl1,hl2,isoh_021[num_layers-1],isoh_021[num_layers-1],sizeiso[num_layers-1], handle, tensor_handle,h1_up,size_h,size_h_new,1);

    cudaHostAlloc((void**)&en,sizeof(float)*size_h_new*size_h_new, 0);

    cudaMemcpy(en,h1_up,sizeof(float)*size_h_new*size_h_new,cudaMemcpyDeviceToHost);
    en[0] = en[0]/(pow(2,L)) + ham_shift *size_h_new;
    cublasIsamin(handle,L-1,min,1,&min_ind); 
    cudaMemcpy(min_sv,min+min_ind,sizeof(float)*1,cudaMemcpyDeviceToHost);
    //t2=clock();
    //times = (double)(t2-t1)/CLOCKS_PER_SEC;
    //cout<<"sweeps :"<<sw<<",energy density: "<<en[0]<<", min_sv:"<<min_sv[0]<<", run-time :"<<times<<endl;
    cudaFree(h1_up);cudaFree(hl1);cudaFree(hl2.mpo1);cudaFree(hl2.mpo2);

  }

}