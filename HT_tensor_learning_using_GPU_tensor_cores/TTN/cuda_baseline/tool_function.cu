#include "head.h"

void printTensor(float *d_des,long m,long n,long l){
	float *des = new float[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(float)*m*n*l,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(long k = 0;k<l;k++){
		for(long i = 0;i<n;i++){
			for(long j = 0;j<m;j++){
				cout<<des[k*m*n+i*m+j]<<" ";
			}
			cout<<endl;
		}
		cout<<"~~~~~~~~~~~~~~~~"<<endl;
	}
	delete[] des;des=nullptr;
}
__global__ void initIdeMat(float *AA,int m){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*m){
    int row = i%m;
    int col = i/m;
    if(row==col){
      AA[col*m+row] = 1;
    }else{
      AA[col*m+row] = 0;
    }
    i+=temp;
  }
  __syncthreads();
}
__global__ void diag_part(float *d_R,float *d_r,int m)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*m)
  {
  	int row = i%m;
    int col = i/m;
    if(row==col){
      d_r[row]=fabs(d_R[col*m+row]);
    }
    i+=temp;
  }
   __syncthreads();
}

__global__ void R_div_r(float *d_R,float *d_r,int m)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*m)
  {
  	int row = i%m;
    int col = i/m;
    if(row==col){
      d_R[col*m+row] = d_R[col*m+row] / d_r[row];
    }else
    {
    	d_R[col*m+row] = 0;
    }
    i+=temp;
  }
   __syncthreads();
}
__global__ void shift_ham(float *h,int n,float ham_shift)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<n*n){
    int row = i%n;
    int col = i/n;
    if(row==col){
      h[col*n+row] = h[col*n+row] -ham_shift;
    }
    i+=temp;
  }
  __syncthreads();
}

__global__ void eye(float *AA,int m)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*m){
    int row = i%m;
    int col = i/m;
    if(row==col){
      AA[col*m+row] = 1;
    }else{
      AA[col*m+row] = 0;
    }
    i+=temp;
  }
  __syncthreads();
}
__global__ void upper(float *A,float *R,int m,int n)
{
    long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;

     while(i<n*n)
    {   
        long row=i/n;
        long col=i%n;
        if(row>=col)    
            R[i]=A[row*m+col];
        else
            R[i]=0;
        i+=temp;        
    }
    __syncthreads();
}

__global__ void mode2(float *A,float *B,long m,long n,long r)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long long temp = blockDim.x*gridDim.x;
  while(i<m*r*n){
    long long row=i/n;
    long long col = i%n;
    long long ge = i/(m*n);
    B[i]=A[(row-ge*m)+(col*m+ge*m*n)];    
    i+=temp;
  }
  __syncthreads();  
}

void ncon_1(float *A_d,float *B_d,float *C_d,vector<int> modeA,vector<int> modeB,vector<int> modeC,unordered_map<int, int64_t> extent,cutensorHandle_t handle)
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

   vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    cutensorTensorDescriptor_t descA;
    cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, CUTENSOR_OP_IDENTITY);

    cutensorTensorDescriptor_t descB;
   cutensorInitTensorDescriptor(&handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, CUTENSOR_OP_IDENTITY);

    cutensorTensorDescriptor_t descC;
   cutensorInitTensorDescriptor( &handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL,/*stride*/
                 typeC, CUTENSOR_OP_IDENTITY);

     uint32_t alignmentRequirementA;
     cutensorGetAlignmentRequirement(&handle,
                  A_d,
                  &descA,
                  &alignmentRequirementA);

     uint32_t alignmentRequirementB;
     cutensorGetAlignmentRequirement(&handle,
                  B_d,
                  &descB,
                  &alignmentRequirementB);

     uint32_t alignmentRequirementC;
     cutensorGetAlignmentRequirement(&handle,
                  C_d,
                  &descC, 
                  &alignmentRequirementC);

    cutensorContractionDescriptor_t desc;
   cutensorInitContractionDescriptor(&handle, 
                 &desc,
                 &descA, modeA.data(), alignmentRequirementA,
                 &descB, modeB.data(), alignmentRequirementB,
                 &descC, modeC.data(), alignmentRequirementC,
                 &descC, modeC.data(), alignmentRequirementC,
                 typeCompute);

    cutensorContractionFind_t find;
    cutensorInitContractionFind( 
                 &handle, &find, 
                 CUTENSOR_ALGO_DEFAULT);

    uint64_t worksize = 0;
    cutensorContractionGetWorkspace(&handle,
                 &desc,
                 &find,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize);

    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    } 

    cutensorContractionPlan_t plan;
    cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize);

    cutensorStatus_t err;
    err = cutensorContraction(&handle,
                                  &plan,
                                  (void*) &alpha, A_d, B_d,
                                  (void*) &beta,  C_d, C_d, 
                                  work, worksize, 0);
        if (err != CUTENSOR_STATUS_SUCCESS)
        {
            cout<<"over"<<endl; 
            printf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
    if (work) cudaFree(work);
}

void gesvdj(float *d_A,float *d_U,float *d_V,float *d_S,int m,int n,cusolverDnHandle_t cusolverH)
{

    float *d_work = NULL;
    int *d_info = NULL; 

    int lwork = 0;
    int info = 0; 

     cudaStream_t stream = NULL;
     gesvdjInfo_t gesvdj_params = NULL;
     float tol = 1.e-7;
     int max_sweeps = 15;
     cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
     cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
     cusolverDnSetStream(cusolverH, stream);
      cusolverDnCreateGesvdjInfo(&gesvdj_params);

      int econ = 1;

    cudaMalloc ((void**)&d_info, sizeof(int));
   
   cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);

   cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);

   cusolverDnSgesvdj_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ, /* econ = 1 for economy size */
        m,    /* nubmer of rows of A, 0 <= m */
        n,    /* number of columns of A, 0 <= n  */
        d_A,  /* m-by-n */
        m,  /* leading dimension of A */
        d_S,  /* min(m,n) */
              /* the singular values in descending order */
        d_U,  /* m-by-m if econ = 0 */
              /* m-by-min(m,n) if econ = 1 */
        m,  /* leading dimension of U, ldu >= max(1,m) */
        d_V,  /* n-by-n if econ = 0  */
              /* n-by-min(m,n) if econ = 1  */
        n,  /* leading dimension of V, ldv >= max(1,n) */
        &lwork,
        gesvdj_params);
    cudaMalloc((void**)&d_work , sizeof(float)*lwork);

   cusolverDnSgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        m,     /* nubmer of rows of A, 0 <= m */
        n,     /* number of columns of A, 0 <= n  */
        d_A,   /* m-by-n */
        m,   /* leading dimension of A */
        d_S,   /* min(m,n)  */               /* the singular values in descending order */
        d_U,   /* m-by-m if econ = 0 */          
        m,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */               /* n-by-min(m,n) if econ = 1  */
        n,   /* leading dimension of V, ldv >= max(1,n) */
        d_work,
        lwork,
        d_info,
        gesvdj_params);
cudaDeviceSynchronize();

    if (d_info) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
}

__global__ void tensorToMode3(float *T1,float *T2,int m,int n,int k){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*n*k){
    int tube = i/(m*n);
    int row = (i-tube*(m*n))%m;
    int col = (i-tube*(m*n))/m;
    T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
    i+=temp;
  }
    __syncthreads();
}
__global__ void tensorToMode1(float *T1,float *T2,int m,int n,int k ){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*n*k){
    int tube = i/(m*n);
    int row = (i-tube*(m*n))%m;
    int col = (i-tube*(m*n))/m;
    T2[tube*m*n+col*m+row] = T1[tube*m*n+col*m+row];
    i+=temp;
  }
  __syncthreads();  
}