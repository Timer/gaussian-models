#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#ifndef MATRIX_HPP
#define MATRIX_HPP

class Matrix;
typedef std::shared_ptr<Matrix> SMatrix;

int _matrix_index_for(int cols, int row, int col) { return row * cols + col; }

class Matrix {
public:
  int rows, cols;
  double *data;

  Matrix(int rows, int cols, bool init) {
    this->rows = rows;
    this->cols = cols;

    data = new double[rows * cols];
    if (init) {
      for (auto i = 0; i < rows * cols; ++i) {
        data[i] = 0;
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

  ~Matrix() { delete[] data; }

  SMatrix transpose() const {
    auto m = std::make_shared<Matrix>(cols, rows, false);
    for (auto n = 0; n < rows * cols; ++n) {
      int i = n / rows;
      int j = n % rows;
      m->data[n] = data[cols * j + i];
    }
    return m;
  }

  SMatrix mean() const {
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

  SMatrix repeat(int makeRows, int copies) const {
    assert(rows == 1);
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

  void print() {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::cout << data[_matrix_index_for(cols, r, c)] << " ";
      }
      std::cout << "\n";
    }
  }
};

SMatrix operator-(const Matrix &m1, const Matrix &m2) {
  assert(m1.rows == m2.rows && m1.cols == m2.cols);
  auto rm = std::make_shared<Matrix>(m1.rows, m1.cols);
  for (auto i = 0; i < m1.rows * m2.cols; ++i) {
    rm->data[i] = m1.data[i] - m2.data[i];
  }
  return rm;
}

SMatrix operator-(const SMatrix &m1, const SMatrix &m2) {
  return *m1.get() - *m2.get();
}

SMatrix operator*(const Matrix &A, const Matrix &B) {
  assert(A.cols == B.rows);
  auto C = std::make_shared<Matrix>(A.rows, B.cols);
  for (auto r = 0; r < C->rows; ++r) {
    for (auto c = 0; c < C->cols; ++c) {
      for (auto inner = 0; inner < A.cols; ++inner) {
        C->data[_matrix_index_for(C->cols, r, c)] +=
            A.data[_matrix_index_for(A.cols, r, inner)] *
            B.data[_matrix_index_for(B.cols, inner, c)];
      }
    }
  }
  return C;
}

SMatrix operator*(const SMatrix &m1, const SMatrix &m2) {
  return *m1.get() * *m2.get();
}

SMatrix operator*(const SMatrix &m1, const Matrix &m2) {
  return *m1.get() * m2;
}

bool operator==(const Matrix &m1, const Matrix &m2) {
  if (m1.rows != m2.rows || m1.cols != m2.cols) {
    return false;
  }
  for (auto i = 0; i < m1.rows * m1.cols; ++i) {
    if (m1.data[i] != m2.data[i]) {
      return false;
    }
  }
  return true;
}

bool operator==(const SMatrix &m1, const Matrix &m2) { return *m1.get() == m2; }

#endif
