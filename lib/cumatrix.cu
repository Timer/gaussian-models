#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>

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
#endif
