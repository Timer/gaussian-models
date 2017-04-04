#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>

template <class T>
__host__ void getLaunchConfiguration(T t, int n, int *blocks, int *threads) {
  cudaOccupancyMaxPotentialBlockSize(blocks, threads, t, 0, n);
  *blocks = (n + *threads - 1) / *threads;
}

__global__ void vec_lgamma(double *a, double *c, const unsigned int n) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = lgamma(a[idx]);
  }
}

__host__ double *cu_lgammed(const int rows, const int cols, double *oData) {
  auto N = rows * cols;
  double *C_accelerate_data = nullptr;
  cudaMalloc(&C_accelerate_data, rows * cols * sizeof(double));
  int blocks, threads;
  getLaunchConfiguration(vec_lgamma, N, &blocks, &threads);
  vec_lgamma<<<blocks, threads>>>(oData, C_accelerate_data, N);
  cudaDeviceSynchronize();
  return C_accelerate_data;
}
#endif
