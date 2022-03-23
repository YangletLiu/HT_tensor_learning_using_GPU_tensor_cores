#ifndef GUARD_func_h
#define GUARD_func_h
typedef float dt;


using std::vector;
using std::unordered_map;

class h2_mpo
{
public:
	float *mpo1=NULL;
	float *mpo2=NULL;
	void init1(int a,int b);
	void init2(int a,int b);
};


void random_tree_tn_uniform(int *Ds,float **iso,int **sizeiso);
void random_isometry(float *A,int m,int n);
void reshape(float *iso,int a,int b,int c);
void opt_tree_energy(float **iso_012,float *h1,h2_mpo h_mpo_2site,int num_sweeps,int itr_l,int verbose,float ham_shift,int **sizeiso);
void printTensor(float *d_des,long m,long n,long l);
__global__ void eye(float *AA,int m);
__global__ void upper(float *A,float *R,int m,int n);
__global__ void diag_part(float *d_R,float *d_r,int m);
__global__ void R_div_r(float *d_R,float *d_r,int m);
__global__ void initIdeMat(float *AA,int m);
__global__ void mode2(float *A,float *B,long m,long n,long r);


//shift_ham  这个应该可以删除这一行
__global__ void shift_ham(float *h,int n,float ham_shift);

void ascend_uniform_op_local(float *h1,h2_mpo h_mpo_2site,float *iso_012,float *iso_021,int *sizeiso,cublasHandle_t handle,cutensorHandle_t tensor_handle,
                             float *h1_new,h2_mpo &h_mpo_2site_new,int size_h,int size_h_new);
void ascend_uniform_op_to_1site(float *h1,h2_mpo h_mpo_2site,float *iso_012,float *iso_021,int *sizeiso,
                                  cublasHandle_t handle,cutensorHandle_t tensor_handle, float *h1_new,int size_h,int size_h_new,int flag);

void _ascend_uniform_op_to_1site_partial(float *h1,h2_mpo h_mpo_2site,float *iso_012,float *iso_021,int *sizeiso,cublasHandle_t handle,
                                         cutensorHandle_t tensor_handle,float *terms_012,float *iso_op_1site_L_021,int size_h,int size_h_new,int flag);
void _ascend_op_2site_to_1site_partial(h2_mpo &h2,float *iso_021,int *sizeiso,
                                         cublasHandle_t handle,cutensorHandle_t tensor_handle,
                                         int size_h,int size_h_new,float *iso_op_2site_012,int flag);
void ascend_op_2site_to_2site(h2_mpo h_mpo_2site,float *iso_012,float *iso_021,int *sizeiso,
                              cublasHandle_t handle,h2_mpo &h_mpo_2site_new,int size_h);
void descend_state_1site(float *states_in,float *iso_012,float *iso_021,int *sizeiso,int *sizein,float *states_out,int *sizeout,
                         cublasHandle_t handle,cutensorHandle_t tensor_handle);

void opt_energy_layer(int i,float **isos_012,float **iso_021,float **states,float *hl1,h2_mpo &hl2,
                      int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,cusolverDnHandle_t cusolverH,int size_h,float *iso_new,float *svs);
void opt_energy_layer_once(int i,float **isos_012,float **isos_021,float *hl1,h2_mpo &hl2,float **states,
                           int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,cusolverDnHandle_t cusolverH,int size_h,float *iso_new,float *svs);
void opt_energy_env(int i,float **isos_012,float **isos_021,float *hl1,h2_mpo &hl2,float **states,
                           int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,int size_h,float *env);

void opt_energy_env_1site(float *iso_012,float *iso_021,float *hl1,h2_mpo &h2,float *states,int *sizeiso,int *size_state,
                          cublasHandle_t handle,cutensorHandle_t tensor_handle,int size_h,float *env,int flag);
void opt_energy_env_2site(int i,float **isos_012,float **isos_021,h2_mpo &hl2,float **states,
                          int **sizeiso,int **size_state,cublasHandle_t handle,cutensorHandle_t tensor_handle,int size_h,float *env2);

void _mpo_with_state(float *iso_012,float *iso_021,int *sizeiso,h2_mpo h2,int size_h2,float *states,int *size_state,
                     cublasHandle_t handle,cutensorHandle_t tensor_handle,float *envL,float *envR);
float _compute_env(int lvl,h2_mpo *ops,bool reflect,int i,float **isos_012,float **isos_021,int **sizeiso,float *states,int *size_state,
                 cublasHandle_t handle,cutensorHandle_t tensor_handle,
                 int size_h,int *size_ops,float *iso_h2R_012,float *iso_h2L_012,float *env);
void _descend_energy_env(float *env,float *ios,int *sizeiso,int size_env,cublasHandle_t handle,cutensorHandle_t tensor_handle);


void ncon_1(float *A_d,float *B_d,float *C_d,vector<int> modeA,vector<int> modeB,vector<int> modeC,unordered_map<int, int64_t> extent,cutensorHandle_t handle);
void gesvdj(float *d_A,float *d_U,float *d_V,float *d_S,int m,int n,cusolverDnHandle_t cusolverH);


__global__ void tensorToMode3(float *T1,float *T2,int m,int n,int k);
__global__ void tensorToMode1(float *T1,float *T2,int m,int n,int k );
#endif