#include <tsqr/blockqr.hpp>
#include <tsqr/tcqr.hpp>
#include <tsqr/tsqr.hpp>
#include<cuda_runtime.h>
#include<cublas_v2.h>

using compute_t = float;

int main()
{
  const bool reorthogonalize = false;
  const auto compute_mode = mtk::qr::compute_mode::fp32_tc_cor;

// size of input matrix
 std::size_t M = 900*900;   //已经运行不动
 std::size_t N = 900;

// allocate input matrix
compute_t *d_a;
cudaMalloc((void**)&d_a, sizeof(compute_t) * N * N);

// allocate output matrices
float *d_r, *d_q;
cudaMalloc((void**)&d_r, sizeof(compute_t) * N * N);
cudaMalloc((void**)&d_q, sizeof(compute_t) * M * N);

// allocate working memory
mtk::qr::buffer<compute_mode, reorthogonalize> buffer;
buffer.allocate(M, N);

// cuBLAS
cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);

// BlockQR
mtk::qr::qr<compute_mode, reorthogonalize>(
	d_q, M,
	d_r, N,
	d_a, M,
	M, N,
	buffer,
	cublas_handle
	);
printf("fdhjsakl\n");
}