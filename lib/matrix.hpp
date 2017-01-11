#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
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
    assert(rows > 0 && cols > 0);
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
    if (rows == 1) {
      return transpose()->mean();
    }
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

  SMatrix diag() const {
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

  double determinant() const {
    assert(rows == cols);
    if (rows == 1)
      return data[0];

    if (rows == 2) {
      return data[_matrix_index_for(cols, 0, 0)] *
                 data[_matrix_index_for(cols, 1, 1)] -
             data[_matrix_index_for(cols, 0, 1)] *
                 data[_matrix_index_for(cols, 1, 0)];
    }

    double det = 0;
    for (int j1 = 0; j1 < rows; ++j1) {
      Matrix m = Matrix(rows - 1, cols - 1);
      for (int i = 1; i < rows; ++i) {
        int j2 = 0;
        for (int j = 0; j < rows; ++j) {
          if (j == j1) {
            continue;
          }
          m.data[_matrix_index_for(cols - 1, i - 1, j2)] =
              data[_matrix_index_for(cols, i, j)];
          ++j2;
          det += std::pow(-1.0, 1.0 + j1 + 1.0) *
                 data[_matrix_index_for(cols, 0, j1)] * m.determinant();
        }
      }
    }
    return det;
  }

  SMatrix cofactor() const {
    assert(rows == cols);
    assert(rows > 1);
    auto b = std::make_shared<Matrix>(rows, cols, false);
    Matrix c(rows - 1, cols - 1);
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        int i1 = 0;
        for (int ii = 0; ii < rows; ++ii) {
          if (ii == i) {
            continue;
          }
          int j1 = 0;
          for (int jj = 0; jj < rows; ++jj) {
            if (jj == j) {
              continue;
            }
            c.data[_matrix_index_for(cols - 1, i1, j1)] =
                data[_matrix_index_for(cols, ii, jj)];
            ++j1;
          }
          ++i1;
        }
        b->data[_matrix_index_for(cols, i, j)] =
            std::pow(-1.0, i + j + 2.0) * c.determinant();
      }
    }
    return b;
  }

  SMatrix scalar(const double &s) {
    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = data[i] * s;
    }
    return m;
  }

  SMatrix power(const double &s) {
    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = std::pow(data[i], s);
    }
    return m;
  }

  SMatrix subtract(const double &s) {
    SMatrix m = std::make_shared<Matrix>(rows, cols, false);
    for (int i = 0; i < rows * cols; ++i) {
      m->data[i] = data[i] - s;
    }
    return m;
  }

  double sumValue() {
    assert(cols == 1);
    double d = 0;
    for (int i = 0; i < rows; ++i) {
      d += data[i];
    }
    return d;
  }

  double value() {
    assert(rows == 1 && cols == 1);
    return data[0];
  }

  SMatrix inverse() const {
    assert(rows == cols);
    return cofactor()->transpose()->scalar(1.0 / determinant());
  }

  bool identity() const {
    assert(rows == cols);
    for (int i = 0; i < rows; ++i) {
      double v = data[_matrix_index_for(cols, i, i)];
      if (std::abs(1 - v) >= std::numeric_limits<double>::epsilon())
        return false;
    }
    return true;
  }

  // TODO: changes need to propogate
  SMatrix list_elems_by_col() const { //(:)
    SMatrix M = std::make_shared<Matrix>(rows * cols, 1);
    for (int c = 0, p = 0; c < cols; ++c) {
      for (int r = 0; r < rows; ++r) {
        M->data[p++] = data[_matrix_index_for(cols, r, c)];
      }
    }
    return M;
  }

  void print() const {
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

SMatrix operator+(const Matrix &A, const Matrix &B) {
  assert(A.rows == B.rows && A.cols == B.cols);
  auto C = std::make_shared<Matrix>(A.rows, A.cols);
  for (auto i = 0; i < A.rows * A.cols; ++i) {
    C->data[i] = A.data[i] + B.data[i];
  }
  return C;
}

SMatrix operator+(const SMatrix &m1, const SMatrix &m2) {
  return *m1.get() + *m2.get();
}

SMatrix operator+(const SMatrix &m1, const Matrix &m2) {
  return *m1.get() + m2;
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
