#include <cassert>
#include <cstdlib>
#include <memory>
#include <ostream>
#include <string>

#ifndef MATRIX_HPP
#define MATRIX_HPP

class Matrix;
typedef std::shared_ptr<Matrix> SMatrix;

int _matrix_index_for(int rows, int row, int col) { return col * rows + row; }

class Matrix {
public:
  int rows, cols;
  double *data;

  Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;

    data = new double[rows * cols];
    for (auto i = 0; i < rows * cols; ++i) {
      data[i] = 0;
    }
  }

  template <typename T, std::size_t rowC, std::size_t colC>
  Matrix(T (&arr)[rowC][colC]) : Matrix(rowC, colC) {
    for (auto r = 0; r < rows; ++r) {
      for (auto c = 0; c < cols; ++c) {
        data[_matrix_index_for(rows, r, c)] = arr[r][c];
      }
    }
  }

  ~Matrix() { delete[] data; }

  SMatrix mean() {
    auto m = std::make_shared<Matrix>(1, cols);
    for (auto r = 0; r < rows; ++r) {
      for (auto c = 0; c < cols; ++c) {
        m->data[c] += data[_matrix_index_for(rows, r, c)];
      }
    }
    for (auto c = 0; c < cols; ++c) {
      m->data[c] /= rows;
    }
    return m;
  }

  SMatrix repeat(int makeRows, int copies) {
    assert(rows == 1);
    auto rm = std::make_shared<Matrix>(makeRows, cols * copies);
    for (auto r = 0; r < makeRows; ++r) {
      for (auto o = 0; o < copies; ++o) {
        for (auto c = 0; c < cols; ++c) {
          rm->data[_matrix_index_for(makeRows, r, o * cols + c)] =
              data[_matrix_index_for(rows, 0, c)];
        }
      }
    }
    return rm;
  }
};

std::ostream &operator<<(std::ostream &stream, const Matrix &matrix) {
  for (auto r = 0; r < matrix.rows; ++r) {
    for (auto c = 0; c < matrix.cols; ++c) {
      stream << matrix.data[_matrix_index_for(matrix.rows, r, c)];
      stream << " ";
    }
    stream << "\n";
  }
}

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

bool operator==(const Matrix &m1, const Matrix &m2) {
  if (m1.rows != m2.rows || m1.cols != m2.cols) {
    return false;
  }
  for (auto i = 0; i < m1.rows * m1.cols; ++i) {
    if (m1.data[i] != m2.data[i])
      return false;
  }
  return true;
}

bool operator==(const SMatrix &m1, const Matrix &m2) { return *m1.get() == m2; }

#endif
