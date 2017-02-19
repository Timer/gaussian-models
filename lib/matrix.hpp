#include <Accelerate/Accelerate.h>
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

#ifndef MATRIX_HPP
#define MATRIX_HPP

#define ACCELERATE_MODE_NONE 0
#define ACCELERATE_MODE_CUDA 1
#define ACCELERATE_MODE_OPENCL 2

#define ACCELERATE_MODE ACCELERATE_MODE_NONE

#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
#include <cublas_v2.h>
#endif

class Matrix;
typedef std::shared_ptr<Matrix> SMatrix;

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
  double *accelerate_data = nullptr;

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

  ~Matrix() { delete[] data; }

  bool shouldAccelerate() {
    if (accelerated)
      return true;

#if ACCELERATE_MODE == ACCELERATE_MODE_NONE
    return false;
#else
    return false;
#endif
  }

  void accelerate() {
    if (accelerated)
      return;

    assert(false);
    accelerated = true;
  }

  void decelerate() {
    if (!accelerated)
      return;

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

  double sumValue() {
    assert(cols == 1);
    decelerate();

    double d = 0;
    for (int i = 0; i < rows; ++i) {
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

  bool identity() {
    assert(rows == cols);
    decelerate();
    for (int i = 0; i < rows; ++i) {
      double v = data[_matrix_index_for(cols, i, i)];
      if (std::abs(1 - v) >= std::numeric_limits<double>::epsilon())
        return false;
    }
    return true;
  }

  // TODO: returned matrix changes need to propogate
  SMatrix list_elems_by_position() { //(:)
    decelerate();
    SMatrix M = std::make_shared<Matrix>(rows * cols, 1);
    for (int p = 1; p <= rows * cols; ++p) {
      M->data[p - 1] = data[_matrix_index_for_position(rows, cols, p)];
    }
    return M;
  }

  // TODO: returned matrix changes need to propogate
  SMatrix list_elems_by_column_position(int col) { //(:, col)
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
  SMatrix row_elems_by_position(int start, int end) { //(start:end)
    decelerate();
    SMatrix M = std::make_shared<Matrix>(1, end - start + 1);
    int index = 0;
    for (int p = start; p <= end; ++p) {
      M->data[index++] = data[_matrix_index_for_position(rows, cols, p)];
    }
    return M;
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

  void set_positions(SMatrix list, const double &value) { //(list) = value
    assert(list->cols == 1);
    decelerate();
    for (int index = 0; index < list->rows; ++index) {
      data[_matrix_index_for_position(rows, cols, list->data[index])] = value;
    }
  }

  void set_positions(SMatrix list, SMatrix source) { // A(list) = B(list)
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
                     double delta) { // A(list) = B(list) + delta
    assert(list->cols == 1);
    assert(rows == source->rows && cols == source->cols);
    decelerate();
    for (int index = 0; index < list->rows; ++index) {
      int p = list->data[index];
      data[_matrix_index_for_position(rows, cols, p)] =
          source->data[_matrix_index_for_position(rows, cols, p)] + delta;
    }
  }

  void set_position(const int &position, const double &value) { //(1) =>
    decelerate();
    data[_matrix_index_for_position(rows, cols, position)] = value;
  }

  double get_position(const int &position) { //(1)
    decelerate();
    return data[_matrix_index_for_position(rows, cols, position)];
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
    auto M = tranA ? cols : rows, N = tranA ? B->rows : B->cols;
    auto J = tranB ? B->cols : B->rows, K = tranB ? rows : cols;
    assert(J == K);

    auto C = std::make_shared<Matrix>(M, N, false);
    if (shouldAccelerate()) {
      accelerate();
      B->accelerate();

#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
      cublasHandle_t handle;
      cublasCreate(&handle); // TODO: store and reuse these handle[s]
      double *C_accelerate_data = nullptr;
      cudaMalloc(&C_accelerate_data, M * N * sizeof(double));
      // We transpose when we don't need to transpose because cublas expecfts
      // col major but we store in row major.
      cublasDgemm(handle, tranA ? CUBLAS_OP_N : CUBLAS_OP_T,
                  tranB ? CUBLAS_OP_N : CUBLAS_OP_T, M, N, K, 1.0,
                  accelerate_data, cols, B->accelerate_data, B->cols, 0.0,
                  C_accelerate_data, C->cols);
      // TODO: force accelerate C (provide memory): C_accelerate_data
      cublasDestroy(handle);
#else
      assert(false);
#endif
    } else {
      decelerate();
      B->decelerate();

      cblas_dgemm(CblasRowMajor, tranA ? CblasTrans : CblasNoTrans,
                  tranB ? CblasTrans : CblasNoTrans, M, N, K, 1.0, data, cols,
                  B->data, B->cols, 0.0, C->data, C->cols);
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

SMatrix load(std::string path) {
  std::ifstream t(path);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  std::vector<std::string> rows = split(str, "\n");
  int colC = split(rows[0], "\t").size();
  SMatrix M = std::make_shared<Matrix>(rows.size(), colC, false);
  for (int r = 0; r < rows.size(); ++r) {
    auto cols = split(rows[r], "\t");
    for (int c = 0; c < colC; ++c) {
      M->data[_matrix_index_for(colC, r, c)] = stod(cols[c]);
    }
  }
  return M;
}

#endif
