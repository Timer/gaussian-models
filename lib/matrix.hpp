#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <streambuf>
#include <string>
#include <vector>

#include "shared.hpp"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
#include <cblas.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cumatrix.hpp"
#endif

class Matrix;
typedef std::shared_ptr<Matrix> SMatrix;
SMatrix operator*(const SMatrix &m1, const SMatrix &m2);

SMatrix eye(int rows, int cols);

inline int _matrix_index_for(int cols, int row, int col) {
  return row * cols + col;
}

inline int _matrix_index_for_position(int rows, int cols, int position) {
  return _matrix_index_for(cols, (position - 1) % rows, (position - 1) / rows);
}

struct Cholesky {
  SMatrix matrix;
  bool error = false;
};

class Matrix {
private:
  bool accelerated = false;
#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
  double *accelerate_data = nullptr;
  void inherit(double *accelerate_data) {
    if (this->accelerate_data != nullptr) {
      std::cout << "You cannot inherit twice." << std::endl;
      throw "You cannot inherit twice.";
    }
    this->accelerated = true;
    this->accelerate_data = accelerate_data;
  }
#elif ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
  cl_mem accelerate_data = nullptr;
  void inherit(cl_mem accelerate_data) {
    if (this->accelerate_data != nullptr) {
      std::cout << "You cannot inherit twice." << std::endl;
      throw "You cannot inherit twice.";
    }
    this->accelerated = true;
    this->accelerate_data = accelerate_data;
  }
#endif

public:
  int rows, cols;
  double *data;

  Matrix(int rows, int cols, bool init) {
    assert(rows >= 0 && cols >= 0);
    this->rows = rows;
    this->cols = cols;
    if (rows == 0 || cols == 0)
      data = nullptr;
    else {
      data = new double[rows * cols];
      if (init) {
        for (auto i = 0; i < rows * cols; ++i) {
          data[i] = 0;
        }
      }
    }
  }

  Matrix(int rows, int cols, double value) {
    assert(rows >= 0 && cols >= 0);
    this->rows = rows;
    this->cols = cols;
    if (rows == 0 || cols == 0)
      data = nullptr;
    else {
      data = new double[rows * cols];
      for (auto i = 0; i < rows * cols; ++i) {
        data[i] = value;
      }
    }
  }

  Matrix(int rows, int cols) : Matrix(rows, cols, true) {}

  template <typename T, std::size_t rowC, std::size_t colC>
  Matrix(T (&arr)[rowC][colC]) : Matrix(rowC, colC) {
    for (auto r = 0; r < rows; ++r) {
      for (auto c = 0; c < cols; ++c) {
        data[_matrix_index_for(cols, r, c)] = arr[r][c];
      }
    }
  }

  Matrix(const Matrix &matrix) : Matrix(matrix.rows, matrix.cols, false) {
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = matrix.data[i];
    }
  }

  ~Matrix() {
    delete[] data;
#if ACCELERATE_MODE == ACCELERATE_MODE_NONE
    return;
#else
    if (accelerate_data == nullptr) {
      return;
    }
#endif
#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
    cudaFree(accelerate_data);
#elif ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
    clReleaseMemObject(accelerate_data);
#else
    assert(false);
#endif
  }

  bool shouldAccelerate(bool linear) {
    if (accelerated) {
      return true;
    }

    const int _1M = 1000000;

#if ACCELERATE_MODE == ACCELERATE_MODE_NONE
    return false;
#elif ACCELERATE_MODE == ACCELERATE_MODE_CUDA
    const int MS_PER_1M_CPU_MULT = 367;
    const int MS_PER_10M_ELEMS = 16;
    return false;
//return linear ? false : rows * cols >= _1M;  // TODO: measure
#elif ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
    const int MS_PER_1M_CPU_MULT = 590;
    const int MS_PER_1M_MULT = 466;
    const int MS_PER_10M_ELEMS = 55;
    return linear ? false : rows * col >= _1M;
#else
    return false;
#endif
  }

  void accelerate() {
    if (accelerated) {
      return;
    }

    auto size = rows * cols * sizeof(data);
#if ACCELERATE_MODE == ACCELERATE_MODE_NONE
    assert(false);
#elif ACCELERATE_MODE == ACCELERATE_MODE_CUDA
    if (accelerate_data == nullptr) {
      cudaMalloc((void **) &accelerate_data, size);
    }
    cudaMemcpy(accelerate_data, data, size, cudaMemcpyHostToDevice);
#elif ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
    if (accelerate_data == nullptr) {
      cl_int err;
      accelerate_data =
          clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, size, NULL, &err);
      err = clEnqueueWriteBuffer(cl_queue, accelerate_data, CL_TRUE, 0, size,
                                 data, 0, NULL, NULL);
    }
#else
    assert(false);
#endif

    accelerated = true;
  }

#if ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
  void checkClError(cl_int res) {
    if (res != CL_SUCCESS) {
      cl_int errs[] = {CL_INVALID_COMMAND_QUEUE, CL_INVALID_CONTEXT, CL_INVALID_MEM_OBJECT, CL_INVALID_VALUE, CL_INVALID_EVENT_WAIT_LIST, CL_MEM_OBJECT_ALLOCATION_FAILURE, CL_OUT_OF_HOST_MEMORY};
      std::string strs[] = {"CL_INVALID_COMMAND_QUEUE", "CL_INVALID_CONTEXT", "CL_INVALID_MEM_OBJECT", "CL_INVALID_VALUE", "CL_INVALID_EVENT_WAIT_LIST", "CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_OUT_OF_HOST_MEMORY"};
      for (auto i = 0; i < 7; ++i) {
        if (res == errs[i]) {
          puts(strs[i].c_str());
          break;
        }
      }
      assert(res == CL_SUCCESS);
    }
  }
#endif

  void decelerate() {
    if (!accelerated) {
      return;
    }

    auto size = rows * cols * sizeof(data);
#if ACCELERATE_MODE == ACCELERATE_MODE_NONE
    assert(false);
#elif ACCELERATE_MODE == ACCELERATE_MODE_CUDA
    cudaMemcpy(data, accelerate_data, size, cudaMemcpyDeviceToHost);
#elif ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
    auto res = clFinish(cl_queue);
    checkClError(res);
    res = clEnqueueReadBuffer(cl_queue, accelerate_data, CL_TRUE, 0, size, data, 0, NULL, NULL);
    checkClError(res);
#else
    assert(false);
#endif

    accelerated = false;
  }

  SMatrix copy(std::vector<int> r, int j0, int j1) {
    decelerate();

    SMatrix X = std::make_shared<Matrix>(r.size(), j1 - j0 + 1, false);
    for (int i = 0; i < r.size(); i++) {
      for (int j = j0; j <= j1; j++) {
        X->data[_matrix_index_for(X->cols, i, j - j0)] =
            data[_matrix_index_for(cols, r[i], j)];
      }
    }
    return X;
  }

  void inplace_set(int row, int col, double value) {
    decelerate();  //TODO: on GPU
    data[_matrix_index_for(cols, row, col)] = value;
  }

  SMatrix transpose() {
    // TODO have an accelerated option
    decelerate();

    auto m = std::make_shared<Matrix>(cols, rows, false);
    for (auto n = 0; n < rows * cols; ++n) {
      int i = n / rows;
      int j = n % rows;
      m->data[n] = data[cols * j + i];
    }
    return m;
  }

  SMatrix mean() {
    if (rows == 1) {
      return transpose()->mean();
    }

    decelerate();
    auto m = std::make_shared<Matrix>(1, cols);
    for (auto r = 0; r < rows; ++r) {
      for (auto c = 0; c < cols; ++c) {
        m->data[c] += data[_matrix_index_for(cols, r, c)];
      }
    }
    for (auto c = 0; c < cols; ++c) {
      m->data[c] /= rows;
    }
    return m;
  }

  SMatrix repeat(int makeRows, int copies) {
    assert(rows == 1);

    decelerate();
    auto rm = std::make_shared<Matrix>(makeRows, cols * copies);
    for (auto r = 0; r < makeRows; ++r) {
      for (auto o = 0; o < copies; ++o) {
        for (auto c = 0; c < cols; ++c) {
          rm->data[_matrix_index_for(rm->cols, r, o * cols + c)] =
              data[_matrix_index_for(cols, 0, c)];
        }
      }
    }
    return rm;
  }

  SMatrix diag() {
    decelerate();

    if (cols == 1) {
      SMatrix m = std::make_shared<Matrix>(rows, rows);
      for (int i = 0; i < rows; ++i) {
        m->data[_matrix_index_for(m->cols, i, i)] = data[i];
      }
      return m;
    } else {
      int min = std::min(rows, cols);
      SMatrix m = std::make_shared<Matrix>(min, 1);
      for (int i = 0; i < min; ++i) {
        m->data[i] = data[_matrix_index_for(cols, i, i)];
      }
      return m;
    }
  }

  SMatrix scalar(const double &s) {
    decelerate();

    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = data[i] * s;
    }
    return m;
  }

  SMatrix power(const double &s) {
    decelerate();

    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = std::pow(data[i], s);
    }
    return m;
  }

  SMatrix add(const double &s) {
    decelerate();

    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = data[i] + s;
    }
    return m;
  }

  SMatrix subtract(const double &s) {
    decelerate();

    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = data[i] - s;
    }
    return m;
  }

  SMatrix abs() {
    decelerate();

    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = std::abs(data[i]);
    }
    return m;
  }

  SMatrix log() {
    decelerate();

    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = std::log(data[i]);
    }
    return m;
  }

  SMatrix floor() {
    decelerate();

    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = std::floor(data[i]);
    }
    return m;
  }

  double multiplyAllValues() {
    decelerate();
    double p = 1;
    for (int i = 0; i < rows * cols; ++i) p *= data[i];
    return p;
  }

  double sumValue() {
    assert(cols == 1);
    decelerate();

    double d = 0;
    for (int i = 0; i < rows; ++i) {
      d += data[i];
    }
    return d;
  }

  double sumAllValue() {
    decelerate();

    double d = 0;
    for (int i = 0; i < rows * cols; ++i) {
      d += data[i];
    }
    return d;
  }

  double trace() {
    assert(rows == cols);
    return diag()->sumValue();
  }

  double value() {
    assert(rows == 1 && cols == 1);
    decelerate();
    return data[0];
  }

private:
  bool isLUNonsingular() {
    assert(rows == cols);
    decelerate();
    for (int j = 0; j < cols; j++) {
      if (data[_matrix_index_for(cols, j, j)] == 0) {
        return false;
      }
    }
    return true;
  }

public:
  SMatrix inverse() {
    assert(rows == cols);
    decelerate();
    auto LU = std::make_shared<Matrix>(*this);
    int m = rows, n = cols;
    std::vector<int> piv;
    for (auto i = 0; i < m; ++i) {
      piv.push_back(i);
    }
    int pivsign = 1;
    // Outer loop.
    for (int j = 0; j < n; j++) {
      // Make a copy of the j-th column to localize references.
      std::vector<double> LUcolj;
      for (int i = 0; i < m; i++) {
        LUcolj.push_back(LU->data[_matrix_index_for(n, i, j)]);
      }

      // Apply previous transformations.
      for (int i = 0; i < m; i++) {
        // Most of the time is spent in the following dot product.

        int kmax = std::min(i, j);
        double s = 0.0;
        for (int k = 0; k < kmax; k++) {
          s += LU->data[_matrix_index_for(n, i, k)] * LUcolj[k];
        }

        LU->data[_matrix_index_for(n, i, j)] = LUcolj[i] -= s;
      }

      // Find pivot and exchange if necessary.

      int p = j;
      for (int i = j + 1; i < m; i++) {
        if (std::abs(LUcolj[i]) > std::abs(LUcolj[p])) {
          p = i;
        }
      }
      if (p != j) {
        for (int k = 0; k < n; k++) {
          double t = LU->data[_matrix_index_for(n, p, k)];
          LU->data[_matrix_index_for(n, p, k)] =
              LU->data[_matrix_index_for(n, j, k)];
          LU->data[_matrix_index_for(n, j, k)] = t;
        }
        int k = piv[p];
        piv[p] = piv[j];
        piv[j] = k;
        pivsign = -pivsign;
      }

      // Compute multipliers.
      if (j < m & LU->data[_matrix_index_for(n, j, j)] != 0.0) {
        for (int i = j + 1; i < m; i++) {
          LU->data[_matrix_index_for(n, i, j)] /=
              LU->data[_matrix_index_for(n, j, j)];
        }
      }
    }
    assert(isLUNonsingular());

    int nx = cols;
    auto X = eye(rows, cols)->copy(piv, 0, nx - 1);

    // Solve L*Y = B(piv,:)
    for (int k = 0; k < n; k++) {
      for (int i = k + 1; i < n; i++) {
        for (int j = 0; j < nx; j++) {
          X->data[_matrix_index_for(X->cols, i, j)] -=
              X->data[_matrix_index_for(X->cols, k, j)] *
              LU->data[_matrix_index_for(LU->cols, i, k)];
        }
      }
    }
    // Solve U*X = Y;
    for (int k = n - 1; k >= 0; k--) {
      for (int j = 0; j < nx; j++) {
        X->data[_matrix_index_for(X->cols, k, j)] /=
            LU->data[_matrix_index_for(LU->cols, k, k)];
      }
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < nx; j++) {
          X->data[_matrix_index_for(X->cols, i, j)] -=
              X->data[_matrix_index_for(X->cols, k, j)] *
              LU->data[_matrix_index_for(LU->cols, i, k)];
        }
      }
    }
    return X;
  }

  bool isIdentity() {
    assert(rows == cols);
    decelerate();
    for (int i = 0; i < rows; ++i) {
      double v = data[_matrix_index_for(cols, i, i)];
      if (std::abs(1 - v) >= /*std::numeric_limits<double>::epsilon()*/ 0.0001)
        return false;
    }
    return true;
  }

  // TODO: returned matrix changes need to propogate
  SMatrix list_elems_by_position() {  //(:)
    decelerate();
    SMatrix M = std::make_shared<Matrix>(rows * cols, 1);
    for (int p = 1; p <= rows * cols; ++p) {
      M->data[p - 1] = data[_matrix_index_for_position(rows, cols, p)];
    }
    return M;
  }

  // TODO: returned matrix changes need to propogate
  SMatrix list_elems_by_row_position(int row) {  //(row, :)
    assert(row > 0 && row <= rows);
    decelerate();
    row -= 1;
    SMatrix M = std::make_shared<Matrix>(1, cols);
    for (int c = 0; c < cols; ++c) {
      M->data[c] = data[_matrix_index_for(cols, row, c)];
    }
    return M;
  }

  // TODO: returned matrix changes need to propogate
  SMatrix list_elems_by_column_position(int col) {  //(:, col)
    assert(col > 0 && col <= cols);
    decelerate();
    col -= 1;
    SMatrix M = std::make_shared<Matrix>(rows, 1);
    for (int r = 0; r < rows; ++r) {
      M->data[r] = data[_matrix_index_for(cols, r, col)];
    }
    return M;
  }

  // TODO: returned matrix changes need to propogate
  SMatrix row_elems_by_position(int start, int end) {  //(start:end)
    decelerate();
    SMatrix M = std::make_shared<Matrix>(1, end - start + 1);
    int index = 0;
    for (int p = start; p <= end; ++p) {
      M->data[index++] = data[_matrix_index_for_position(rows, cols, p)];
    }
    return M;
  }

  template <class T>
  std::vector<T> asVector() {
    assert(rows == 1 || cols == 1);
    decelerate();

    std::vector<T> v;
    for (int i = 0; i < rows * cols; ++i) {
      v.push_back(data[i]);
    }
    return std::move(v);
  }

  SMatrix create_sz() {
    decelerate();
    SMatrix r = std::make_shared<Matrix>(rows, 1);
    for (int i = 0; i < rows; ++i) {
      int max = 2;
      for (int temp = 1, j = 0; j < cols; ++j) {
        temp = data[_matrix_index_for(cols, i, j)];
        max = max < temp ? temp : max;
      }
      r->data[i] = max;
    }
    return r;
  }

  SMatrix extract_indices(int row_start, int row_stop, int col_start, int col_stop) {
    decelerate();
    SMatrix m = std::make_shared<Matrix>(row_stop - row_start, col_stop - col_start, false);
    for (int r = 0; r < m->rows; ++r) {
      for (int c = 0; c < m->cols; ++c) {
        m->data[_matrix_index_for(m->cols, r, c)] = data[_matrix_index_for(cols, row_start + r, col_start + c)];
      }
    }
    return m;
  }

  SMatrix extract_list_index(std::vector<int> _rows, int col_start, int col_stop) {
    decelerate();
    SMatrix m = std::make_shared<Matrix>(_rows.size(), col_stop - col_start, false);
    for (int r = 0; r < m->rows; ++r) {
      for (int c = 0; c < m->cols; ++c) {
        m->data[_matrix_index_for(m->cols, r, c)] = data[_matrix_index_for(cols, _rows[r], col_start + c)];
      }
    }
    return m;
  }

  SMatrix extract_list_index(int row_start, int row_stop, std::vector<int> _cols) {
    decelerate();
    SMatrix m = std::make_shared<Matrix>(row_stop - row_start, _cols.size(), false);
    for (int r = 0; r < m->rows; ++r) {
      for (int c = 0; c < m->cols; ++c) {
        m->data[_matrix_index_for(m->cols, r, c)] = data[_matrix_index_for(cols, row_start + r, _cols[c])];
      }
    }
    return m;
  }

  void set_list_index(double value, std::vector<int> _rows, int col_start, int col_stop) {
    decelerate();
    const int cRows = _rows.size(), cCols = col_stop - col_start;
    for (int r = 0; r < cRows; ++r) {
      for (int c = 0; c < cCols; ++c) {
        data[_matrix_index_for(cols, _rows[r], col_start + c)] = value;
      }
    }
  }

  void set_list_index(double value, int row_start, int row_stop, std::vector<int> _cols) {
    decelerate();
    const int cRows = row_stop - row_start, cCols = _cols.size();
    for (int r = 0; r < cRows; ++r) {
      for (int c = 0; c < cCols; ++c) {
        data[_matrix_index_for(cols, row_start + r, _cols[c])] = value;
      }
    }
  }

  std::vector<int> adjacency_matrix_parents(const int col) {
    decelerate();
    std::vector<int> l;
    SMatrix sub = extract_indices(0, rows, col, col + 1);
    for (int i = 0; i < rows; ++i) {
      int val = sub->data[i];
      assert(val == 0 || val == 1);
      if (val) l.push_back(i);
    }
    return std::move(l);
  }

  SMatrix concat_rows(SMatrix appendMatrix, bool parallel_enabled) {
    decelerate();
    SMatrix m = std::make_shared<Matrix>(rows + appendMatrix->rows, cols, false);
#pragma omp parallel for if (parallel_enabled) collapse(2)
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        m->data[_matrix_index_for(m->cols, r, c)] = data[_matrix_index_for(cols, r, c)];
      }
    }
#pragma omp parallel for if (parallel_enabled) collapse(2)
    for (int r = 0; r < appendMatrix->rows; ++r) {
      for (int c = 0; c < appendMatrix->cols; ++c) {
        m->data[_matrix_index_for(m->cols, rows + r, c)] = appendMatrix->data[_matrix_index_for(appendMatrix->cols, r, c)];
      }
    }
    return m;
  }

  SMatrix sum_n_cols(int wCols) {
    decelerate();
    SMatrix m = std::make_shared<Matrix>((rows * cols) / wCols, 1);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i % m->rows] += data[i];
    }
    return m;
  }

#if ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
  const char *lgammaKernelSource =
      "\n"
      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable    \n"
      "__kernel void vec_lgamma(__global double *a,     \n"
      "                         __global double *c,     \n"
      "                         const unsigned int n)   \n"
      "{                                                \n"
      "  const unsigned int id = get_global_id(0);      \n"
      "  if (id < n)                                    \n"
      "    c[id] = lgamma(a[id]);                       \n"
      "}                                                \n"
      "\n";
#endif

  SMatrix lgammaed() {
    auto C = std::make_shared<Matrix>(rows, cols, false);
#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
    if (false) {
#else
    if (shouldAccelerate(true)) {
#endif
      accelerate();

#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
      C->inherit(cu_lgammed(rows, cols, accelerate_data));
#elif ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
      auto N = rows * cols;
      cl_int err;
      cl_event event;
      auto C_accelerate_data = clCreateBuffer(
          cl_ctx, CL_MEM_WRITE_ONLY, rows * cols * sizeof(double), nullptr, &err);
      checkClError(err);
      auto program = clCreateProgramWithSource(cl_ctx, 1, (const char **) &lgammaKernelSource, NULL, &err);
      clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
      auto kernel = clCreateKernel(program, "vec_lgamma", &err);  //TODO: reuse kernel (& program)
      err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &accelerate_data);
      err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &C_accelerate_data);
      err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &N);
      size_t localSize = 64;
      size_t globalSize = std::ceil(N / (float) localSize) * localSize;
      err = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
      clFinish(cl_queue);
      clReleaseProgram(program);
      clReleaseKernel(kernel);
      C->inherit(C_accelerate_data);
#else
    assert(false);
#endif
    } else {
      decelerate();
      for (auto i = 0; i < rows * cols; ++i) {
        C->data[i] = lgamma(data[i]);
      }
    }
    return C;
  }

  void mk_stochastic(SMatrix ns) {
    decelerate();
    int dim = ns->data[(ns->rows * ns->cols) - 1];
    int index_count = rows * cols;
    assert(index_count % dim == 0);
    SMatrix div = sum_n_cols(dim);
    for (int i = 0; i < index_count; ++i) {
      data[i] = data[i] / div->data[i % (index_count / dim)];
    }
  }

  SMatrix find_positions(const double &alpha, const bool greater,
                         const bool equal) {
    decelerate();
    std::vector<int> positions;
    for (int p = 1; p <= rows * cols; ++p) {
      auto value = data[_matrix_index_for_position(rows, cols, p)];
      if (greater) {
        if (equal) {
          if (value >= alpha)
            positions.push_back(p);
        } else {
          if (value > alpha)
            positions.push_back(p);
        }
      } else {
        if (equal) {
          if (value <= alpha)
            positions.push_back(p);
        } else {
          if (value < alpha)
            positions.push_back(p);
        }
      }
    }
    SMatrix M = std::make_shared<Matrix>(positions.size(), 1);
    for (int i = 0; i < M->rows; ++i)
      M->data[i] = positions[i];
    return M;
  }

  void set_positions(SMatrix list, const double &value) {  //(list) = value
    assert(list->cols == 1);
    decelerate();
    for (int index = 0; index < list->rows; ++index) {
      data[_matrix_index_for_position(rows, cols, list->data[index])] = value;
    }
  }

  void set_positions(SMatrix list, SMatrix source) {  // A(list) = B(list)
    assert(list->cols == 1);
    assert(rows == source->rows && cols == source->cols);
    decelerate();
    for (int index = 0; index < list->rows; ++index) {
      int p = list->data[index];
      data[_matrix_index_for_position(rows, cols, p)] =
          source->data[_matrix_index_for_position(rows, cols, p)];
    }
  }

  void set_positions(SMatrix list, SMatrix source,
                     double delta) {  // A(list) = B(list) + delta
    assert(list->cols == 1);
    assert(rows == source->rows && cols == source->cols);
    decelerate();
    for (int index = 0; index < list->rows; ++index) {
      int p = list->data[index];
      data[_matrix_index_for_position(rows, cols, p)] =
          source->data[_matrix_index_for_position(rows, cols, p)] + delta;
    }
  }

  void set_position(const int &position, const double &value) {  //(1) =>
    decelerate();
    data[_matrix_index_for_position(rows, cols, position)] = value;
  }

  double get_position(const int &position) {  //(1)
    decelerate();
    return data[_matrix_index_for_position(rows, cols, position)];
  }

  void set_position(const int &r_position, const int &c_position, const double &value) {  //(1, 1) =>
    decelerate();
    data[_matrix_index_for(cols, r_position - 1, c_position - 1)] = value;
  }

  double get_position(const int &r_position, const int &c_position) {  //(1, 1)
    decelerate();
    return data[_matrix_index_for(cols, r_position - 1, c_position - 1)];
  }

  SMatrix triu(int kdiag) {
    decelerate();

    auto M = std::make_shared<Matrix>(rows, cols, true);
    for (int r = 0; r < rows; ++r) {
      for (int c = kdiag + r; c < cols; ++c) {
        M->data[_matrix_index_for(cols, r, c)] =
            data[_matrix_index_for(cols, r, c)];
      }
    }
    return M;
  }

  Cholesky cholesky() {
    assert(rows == cols);
    decelerate();

    Cholesky ch;
    ch.matrix = std::make_shared<Matrix>(rows, cols);

    bool isspd = true;
    auto n = rows;
    for (int j = 0; j < n && isspd; j++) {
      double d = 0.0;
      for (int k = 0; k < j && isspd; k++) {
        double s = 0.0;
        for (int i = 0; i < k && isspd; i++) {
          s += ch.matrix->data[_matrix_index_for(cols, i, k)] *
               ch.matrix->data[_matrix_index_for(cols, i, j)];
        }
        ch.matrix->data[_matrix_index_for(cols, k, j)] = s =
            (data[_matrix_index_for(cols, j, k)] - s) /
            ch.matrix->data[_matrix_index_for(cols, k, k)];
        d = d + s * s;
        isspd = isspd & (data[_matrix_index_for(cols, k, j)] ==
                         data[_matrix_index_for(cols, j, k)]);
      }
      d = data[_matrix_index_for(cols, j, j)] - d;
      isspd = isspd & (d > 0.0);
      ch.matrix->data[_matrix_index_for(cols, j, j)] =
          std::sqrt(std::max(d, 0.0));
      for (int k = j + 1; k < n; k++) {
        ch.matrix->data[_matrix_index_for(cols, k, j)] = 0.0;
      }
    }
    ch.error = !isspd;
    return ch;
  }

  SMatrix multiply(const SMatrix B) { return multiply(B, false, false); }

  SMatrix multiply(const SMatrix B, const bool tranA, const bool tranB) {
    auto M = tranA ? cols : rows, N = tranB ? B->rows : B->cols;
    auto J = tranB ? B->cols : B->rows, K = tranA ? rows : cols;
    assert(J == K);

    auto C = std::make_shared<Matrix>(M, N, false);
    if (shouldAccelerate(false)) {
      accelerate();
      B->accelerate();

#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
      cublasHandle_t handle;
      cublasCreate(&handle);  // TODO: store and reuse these handle[s]
      double *C_accelerate_data = nullptr;
      cudaMalloc((void **) &C_accelerate_data, M * N * sizeof(double));
      // We transpose when we don't need to transpose because cublas expects
      // col major but we store in row major.
      const auto alpha = 1.0, beta = 0.0;
      cublasDgemm(handle, tranA ? CUBLAS_OP_N : CUBLAS_OP_T,
                  tranB ? CUBLAS_OP_N : CUBLAS_OP_T, M, N, K, &alpha,
                  accelerate_data, cols, B->accelerate_data, B->cols, &beta,
                  C_accelerate_data, C->cols);
      C->inherit(C_accelerate_data);
      cublasDestroy(handle);
#elif ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
      cl_int err;
      cl_event event;
      auto C_accelerate_data = clCreateBuffer(
          cl_ctx, CL_MEM_READ_WRITE, M * N * sizeof(double), NULL, &err);
      err = clblasDgemm(clblasRowMajor, tranA ? clblasTrans : clblasNoTrans,
                        tranB ? clblasTrans : clblasNoTrans, M, N, K, 1.0,
                        accelerate_data, 0, cols, B->accelerate_data, 0,
                        B->cols, 0.0, C_accelerate_data, 0, C->cols, 1,
                        &cl_queue, 0, NULL, &event);
      if (err != clblasSuccess) {
        std::cout << "Could not execute dgemm." << std::endl;
        throw "Could not execute dgemm.";
      }
      err = clWaitForEvents(1, &event);
      C->inherit(C_accelerate_data);
#else
    assert(false);
#endif
    } else {
      decelerate();
      B->decelerate();
#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
      if (tranA) {
        return transpose()->multiply(B, false, tranB);
      } else if (tranB) {
        return multiply(B->transpose(), false, false);
      }
      double *Bcolj = new double[cols];
      for (int j = 0; j < B->cols; j++) {
        for (int k = 0; k < cols; k++) {
          Bcolj[k] = B->data[_matrix_index_for(B->cols, k, j)];
        }
        for (int i = 0; i < rows; i++) {
          double s = 0;
          for (int k = 0; k < cols; k++) {
            s += data[_matrix_index_for(cols, i, k)] * Bcolj[k];
          }
          C->data[_matrix_index_for(C->cols, i, j)] = s;
        }
      }
      delete[] Bcolj;
#else
      cblas_dgemm(CblasRowMajor, tranA ? CblasTrans : CblasNoTrans,
                  tranB ? CblasTrans : CblasNoTrans, M, N, K, 1.0, data, cols,
                  B->data, B->cols, 0.0, C->data, C->cols);
#endif
    }
    return C;
  }

  void print() {
    decelerate();

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::cout << data[_matrix_index_for(cols, r, c)] << " ";
      }
      std::cout << "\n";
    }
  }

  void noop() {
  }

  void save(std::string fileName) {
    decelerate();
    std::ofstream file;
    file.open(fileName, std::ios::trunc);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols - 1; ++c) {
        file << data[_matrix_index_for(cols, r, c)] << ",";
      }
      file << data[_matrix_index_for(cols, r, cols - 1)] << "\n";
    }
    file.close();
  }
};

SMatrix eye(int rows, int cols) {
  auto M = std::make_shared<Matrix>(rows, cols);
  for (int p = 0; p < std::min(rows, cols); ++p) {
    M->data[_matrix_index_for(cols, p, p)] = 1;
  }
  return M;
}

SMatrix operator-(const SMatrix &m1, const SMatrix &m2) {
  assert(m1->rows == m2->rows && m1->cols == m2->cols);
  m1->decelerate();
  m2->decelerate();

  auto rm = std::make_shared<Matrix>(m1->rows, m1->cols, false);
  for (auto i = 0; i < m1->rows * m2->cols; ++i) {
    rm->data[i] = m1->data[i] - m2->data[i];
  }
  return rm;
}

SMatrix operator*(const SMatrix &m1, const SMatrix &m2) {
  return m1->multiply(m2);
}

SMatrix operator+(const SMatrix &A, const SMatrix &B) {
  assert(A->rows == B->rows && A->cols == B->cols);
  A->decelerate();
  B->decelerate();
  auto C = std::make_shared<Matrix>(A->rows, A->cols, false);
  for (auto i = 0; i < A->rows * A->cols; ++i) {
    C->data[i] = A->data[i] + B->data[i];
  }
  return C;
}

bool operator==(const SMatrix m1, const SMatrix m2) {
  if (m1->rows != m2->rows || m1->cols != m2->cols) {
    return false;
  }
  m1->decelerate();
  m2->decelerate();
  for (auto i = 0; i < m1->rows * m1->cols; ++i) {
    if (m1->data[i] != m2->data[i]) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> split(const std::string &s, const std::string &delim,
                               const bool keep_empty = false) {
  std::vector<std::string> result;
  if (delim.empty()) {
    result.push_back(s);
    return result;
  }
  std::string::const_iterator substart = s.begin(), subend;
  while (true) {
    subend = std::search(substart, s.end(), delim.begin(), delim.end());
    std::string temp(substart, subend);
    if (keep_empty || !temp.empty()) {
      result.push_back(temp);
    }
    if (subend == s.end()) {
      break;
    }
    substart = subend + delim.size();
  }
  return result;
}

SMatrix load(std::string path, bool transposed) {
  std::ifstream t(path);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  std::vector<std::string> rows = split(str, "\n");
  auto splitChar = "\t";
  if (rows[0].find(splitChar) == std::string::npos) {
    splitChar = ",";
  }
  int colC = split(rows[0], splitChar).size();
  SMatrix M = std::make_shared<Matrix>(rows.size(), colC, false);
  for (int r = 0; r < rows.size(); ++r) {
    auto cols = split(rows[r], splitChar);
    for (int c = 0; c < colC; ++c) {
      M->data[_matrix_index_for(colC, r, c)] = stod(cols[c]);
    }
  }
  if (transposed) return M->transpose();
  return M;
}

#endif
