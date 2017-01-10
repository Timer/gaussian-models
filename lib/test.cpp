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

  int tranA[3][2] = {{1, 4}, {2, 5}, {3, 6}};
  assert(m1.transpose() == Matrix(tranA));

  int mulA[3][3] = {{17, 22, 27}, {22, 29, 36}, {27, 36, 45}};
  assert(m1.transpose() * m1 == Matrix(mulA));
  return 0;
}
