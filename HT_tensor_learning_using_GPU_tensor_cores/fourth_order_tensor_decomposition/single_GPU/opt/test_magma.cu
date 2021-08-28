// This is a simple standalone example. See README.txt

#include <stdio.h>
#include <stdlib.h>

//#include "cublas_v2.h"     // if you need CUBLAS v2, include before magma.h
#include "magma.h"
//#include "magma_lapack.h"  // if you need BLAS & LAPACK


// ------------------------------------------------------------
int main( int argc, char** argv )
{
    magma_init();
    float *A = new float[3*3];
    float *s = new float[3];
    float *U = new float[3*3];
    float *VT = new float[3*3];

    for(unsigned i = 0; i < 3*3; ++i) {
      /* code */
      A[i] = rand()*1.0/(RAND_MAX*1.0);
    }
     float *d_A,*d_s,*d_U,*d_VT;
     cudaMalloc((void**)&d_A,sizeof(float)*3*3);
     cudaMalloc((void**)&d_s,sizeof(float)*3);
     cudaMalloc((void**)&d_U,sizeof(float)*3*3);
     cudaMalloc((void**)&d_VT,sizeof(float)*3*3);

     cudaMemcpy(d_A,A,sizeof(float)*3*3,cudaMemcpyHostToDevice);

     magma_vec_t jobu=MagmaAllVec;
     magma_vec_t jobvt = MagmaAllVec;
     float work[1];
     float *hwork;
     int lwork = -1;
     int info;
     magma_sgesvd (jobu,
                   jobvt,
                    3,
                    3,
                    NULL,
                    3,
                    NULL,
                    NULL,
                    3,
                    NULL,
                    3,
                    work,
                    lwork,
                    &info 
                    ); 
    cudaDeviceSynchronize();

     magma_smalloc_pinned( &hwork, work[0] );
     int w;w=work[0]-1;
    printf("fen pei kong jian\n");
    magma_sgesvd (jobu,
                   jobvt,
                    3,
                    3,
                    d_A,
                    3,
                    d_s,
                    d_U,
                    3,
                    d_VT,
                    3,
                    hwork,
                    w,
                    &info 
                    );
    cudaDeviceSynchronize();
    printf("f-----n\n");
    cudaMemcpy(s,d_s,sizeof(float)*3,cudaMemcpyDeviceToHost);
    for(unsigned i = 0; i < 3; ++i) {
      /* code */
      printf(" %.2f ,",s[i]);
    }
    printf("\n");

    magma_finalize();
    return 0;
}