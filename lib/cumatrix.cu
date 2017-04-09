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

__host__ double *cu_lgammed(const int rows, const int cols, double *iData) {
  auto N = rows * cols;
  double *C_accelerate_data = nullptr;
  cudaMalloc((void **) &C_accelerate_data, rows * cols * sizeof(double));
  int blocks, threads;
  getLaunchConfiguration(vec_lgamma, N, &blocks, &threads);
  vec_lgamma<<<blocks, threads>>>(iData, C_accelerate_data, N);
  cudaDeviceSynchronize();
  return C_accelerate_data;
}

__global__ void vec_add(double *a, double *b, double *c, const unsigned int n) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void vec_sub(double *a, double *b, double *c, const unsigned int n) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] - b[idx];
  }
}

__host__ double *cu_add(const int rows, const int cols, double *m1, double *m2) {
  auto N = rows * cols;
  double *C_accelerate_data = nullptr;
  cudaMalloc((void **) &C_accelerate_data, rows * cols * sizeof(double));
  int blocks, threads;
  getLaunchConfiguration(vec_add, N, &blocks, &threads);
  vec_add<<<blocks, threads>>>(m1, m2, C_accelerate_data, N);
  cudaDeviceSynchronize();
  return C_accelerate_data;
}

__host__ double *cu_sub(const int rows, const int cols, double *m1, double *m2) {
  auto N = rows * cols;
  double *C_accelerate_data = nullptr;
  cudaMalloc((void **) &C_accelerate_data, rows * cols * sizeof(double));
  int blocks, threads;
  getLaunchConfiguration(vec_sub, N, &blocks, &threads);
  vec_sub<<<blocks, threads>>>(m1, m2, C_accelerate_data, N);
  cudaDeviceSynchronize();
  return C_accelerate_data;
}
#endif
