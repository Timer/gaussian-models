#include "matrix.hpp"
#include <cassert>
#include <iostream>

int main(int argc, char *argv[]) {
  int sampleA[2][3] = {{1, 2, 3}, {4, 5, 6}};
  double meanA[1][3] = {{2.5, 3.5, 4.5}};
  auto m1 = Matrix(sampleA), mean1 = Matrix(meanA);
  assert(m1.mean() == mean1);
  double r1[2][3] = {{2.5, 3.5, 4.5}, {2.5, 3.5, 4.5}};
  auto rep1 = Matrix(r1);
  assert(mean1.repeat(2, 1) == rep1);
  double sub1[2][3] = {{-1.5, -1.5, -1.5}, {1.5, 1.5, 1.5}};
  assert((m1 - rep1) == Matrix(sub1));

  int meanB[3][1] = {{1}, {2}, {3}};
  int mB[1][1] = {{2}};
  assert(Matrix(meanB).mean() == Matrix(mB));
  int meanC[1][3] = {{1, 2, 3}};
  int mC[1][1] = {{2}};
  assert(Matrix(meanC).mean() == Matrix(mC));

  int tranA[3][2] = {{1, 4}, {2, 5}, {3, 6}};
  assert(m1.transpose() == Matrix(tranA));

  int mulA[3][3] = {{17, 22, 27}, {22, 29, 36}, {27, 36, 45}};
  assert(m1.transpose() * m1 == Matrix(mulA));

  int diag1[3][1] = {{1}, {2}, {3}};
  int mdiag1[3][3] = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
  assert(Matrix(diag1).diag() == Matrix(mdiag1));
  int mdiag2[3][7] = {{1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14},
                      {15, 16, 17, 18, 19, 20, 21}};
  int diag2[3][1] = {{1}, {9}, {17}};
  assert(Matrix(mdiag2).diag() == Matrix(diag2));
  int mdiag3[6][3] = {{1, 2, 3},    {4, 5, 6},    {7, 8, 9},
                      {10, 11, 12}, {13, 14, 15}, {16, 17, 18}};
  int diag3[3][1] = {{1}, {5}, {9}};
  assert(Matrix(mdiag3).diag() == Matrix(diag3));

  int mDet1[3][3] = {{1, 0, 2}, {-1, 5, 0}, {0, 3, -9}};
  assert(Matrix(mDet1).determinant() == -51);
  return 0;
}
