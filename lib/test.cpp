#include <cassert>
#include <chrono>
#include <iostream>
#include "matrix.hpp"
#include "shared.hpp"
#include "sparfun.hpp"

inline std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

inline int to_seconds(std::chrono::time_point<std::chrono::high_resolution_clock> t1, std::chrono::time_point<std::chrono::high_resolution_clock> t2) {
  std::chrono::duration<double> diff = t2 - t1;
  return diff.count();
}

inline int to_milliseconds(std::chrono::time_point<std::chrono::high_resolution_clock> t1, std::chrono::time_point<std::chrono::high_resolution_clock> t2) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
}

int main(int argc, char *argv[]) {
  event_start();
  int sampleA[2][3] = {{1, 2, 3}, {4, 5, 6}};
  double meanA[1][3] = {{2.5, 3.5, 4.5}};
  auto m1 = std::make_shared<Matrix>(sampleA),
       mean1 = std::make_shared<Matrix>(meanA);
  assert(m1->mean() == mean1);
  double r1[2][3] = {{2.5, 3.5, 4.5}, {2.5, 3.5, 4.5}};
  auto rep1 = std::make_shared<Matrix>(r1);
  assert(mean1->repeat(2, 1) == rep1);
  double sub1[2][3] = {{-1.5, -1.5, -1.5}, {1.5, 1.5, 1.5}};
  assert((m1 - rep1) == std::make_shared<Matrix>(sub1));

  int meanB[3][1] = {{1}, {2}, {3}};
  int mB[1][1] = {{2}};
  assert(std::make_shared<Matrix>(meanB)->mean() ==
         std::make_shared<Matrix>(mB));
  int meanC[1][3] = {{1, 2, 3}};
  int mC[1][1] = {{2}};
  assert(std::make_shared<Matrix>(meanC)->mean() ==
         std::make_shared<Matrix>(mC));

  int tranA[3][2] = {{1, 4}, {2, 5}, {3, 6}};
  assert(m1->transpose() == std::make_shared<Matrix>(tranA));

  int mulA[3][3] = {{17, 22, 27}, {22, 29, 36}, {27, 36, 45}};
  assert(m1->transpose() * m1 == std::make_shared<Matrix>(mulA));

  int diag1[3][1] = {{1}, {2}, {3}};
  int mdiag1[3][3] = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
  assert(std::make_shared<Matrix>(diag1)->diag() ==
         std::make_shared<Matrix>(mdiag1));
  int mdiag2[3][7] = {{1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14},
                      {15, 16, 17, 18, 19, 20, 21}};
  int diag2[3][1] = {{1}, {9}, {17}};
  assert(std::make_shared<Matrix>(mdiag2)->diag() ==
         std::make_shared<Matrix>(diag2));
  int mdiag3[6][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}};
  int diag3[3][1] = {{1}, {5}, {9}};
  assert(std::make_shared<Matrix>(mdiag3)->diag() ==
         std::make_shared<Matrix>(diag3));

  int mDet1[3][3] = {{1, 0, 2}, {-1, 5, 0}, {0, 3, -9}};
  int abs_mDet1[3][3] = {{1, 0, 2}, {1, 5, 0}, {0, 3, 9}};
  assert(std::make_shared<Matrix>(mDet1)->abs() ==
         std::make_shared<Matrix>(abs_mDet1));

  int mDet2[3][3] = {{1, 2, 0}, {-1, 1, 1}, {1, 2, 3}};
  assert((std::make_shared<Matrix>(mDet2)->inverse() *
          std::make_shared<Matrix>(mDet2))
             ->isIdentity());
  int sumA[2][3] = {{1, 2, 3}, {4, 5, 6}}, sumB[2][3] = {{4, 5, 6}, {1, 2, 3}};
  int mSum1[2][3] = {{5, 7, 9}, {5, 7, 9}};
  int mSub1[2][3] = {{-3, -3, -3}, {3, 3, 3}};
  assert(std::make_shared<Matrix>(sumA) + std::make_shared<Matrix>(sumB) ==
         std::make_shared<Matrix>(mSum1));
  assert(std::make_shared<Matrix>(sumA) - std::make_shared<Matrix>(sumB) ==
         std::make_shared<Matrix>(mSub1));

  int listCols1[6][1] = {{1}, {4}, {2}, {5}, {3}, {6}};
  assert(m1->list_elems_by_position() == std::make_shared<Matrix>(listCols1));

  int doubleA[2][3] = {{2, 4, 6}, {8, 10, 12}};
  assert(m1->scalar(2) == std::make_shared<Matrix>(doubleA));
  int subA[2][3] = {{0, 1, 2}, {3, 4, 5}};
  assert(m1->subtract(1) == std::make_shared<Matrix>(subA));
  int addA[2][3] = {{2, 3, 4}, {5, 6, 7}};
  assert(m1->add(1) == std::make_shared<Matrix>(addA));
  int powA[2][3] = {{1, 4, 9}, {16, 25, 36}};
  assert(m1->power(2) == std::make_shared<Matrix>(powA));
  int sM[3][1] = {{1}, {2}, {3}};
  assert(std::make_shared<Matrix>(sM)->sumValue() == 6);
  int fsM[3][2] = {{1, 2}, {3, 4}, {5, 6}};
  int singleM[1][1] = {{5}};
  assert(std::make_shared<Matrix>(singleM)->value() == 5);

  // TODO: test set_position
  assert(m1->get_position(1) == 1);
  assert(m1->get_position(2) == 4);
  assert(m1->get_position(3) == 2);
  assert(m1->get_position(4) == 5);
  assert(m1->get_position(5) == 3);
  assert(m1->get_position(6) == 6);

  int pM1[3][3] = {{1, -1, -1}, {-1, 2, 0}, {-1, 0, 3}};
  int cM1[3][3] = {{1, -1, -1}, {0, 1, -1}, {0, 0, 1}};
  assert(std::make_shared<Matrix>(pM1)->cholesky().matrix ==
         std::make_shared<Matrix>(cM1));

  int nPD1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  assert(std::make_shared<Matrix>(nPD1)->cholesky().error);

  int nPD2[3][3] = {{1, 2, 3}, {2, 3, 3}, {3, 3, 3}};
  assert(std::make_shared<Matrix>(nPD2)->cholesky().error);

  int lT1[2][2] = {{1, 2}, {3, 4}};
  double lmT2[2][2] = {{std::log(1), std::log(2)}, {std::log(3), std::log(4)}};
  assert(std::make_shared<Matrix>(lT1)->log() ==
         std::make_shared<Matrix>(lmT2));

  int rebp1[1][4] = {{4, 7, 2, 5}};
  assert(std::make_shared<Matrix>(nPD1)->row_elems_by_position(2, 5) ==
         std::make_shared<Matrix>(rebp1));

  int setex[3][1] = {{1}, {4}, {8}};
  int exset[3][3] = {{0, 0, 3}, {4, 5, 0}, {7, 8, 9}};
  auto tempM = std::make_shared<Matrix>(nPD1);
  tempM->set_positions(std::make_shared<Matrix>(setex), 0);
  assert(tempM == std::make_shared<Matrix>(exset));

  int find1[3][1] = {{1}, {4}, {7}};
  assert(std::make_shared<Matrix>(nPD1)->find_positions(4, false, false) ==
         std::make_shared<Matrix>(find1));
  int find2[4][1] = {{1}, {2}, {4}, {7}};
  assert(std::make_shared<Matrix>(nPD1)->find_positions(4, false, true) ==
         std::make_shared<Matrix>(find2));
  int find3[5][1] = {{3}, {5}, {6}, {8}, {9}};
  assert(std::make_shared<Matrix>(nPD1)->find_positions(4, true, false) ==
         std::make_shared<Matrix>(find3));
  int find4[6][1] = {{2}, {3}, {5}, {6}, {8}, {9}};
  assert(std::make_shared<Matrix>(nPD1)->find_positions(4, true, true) ==
         std::make_shared<Matrix>(find4));

  int nPD3[3][3] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
  auto tmLSM = std::make_shared<Matrix>(nPD1);
  auto list = tmLSM->find_positions(4, false, false);
  int nPD4[3][3] = {{8, 7, 6}, {4, 5, 6}, {7, 8, 9}};
  tmLSM->set_positions(list, std::make_shared<Matrix>(nPD3)->subtract(1));
  assert(tmLSM == std::make_shared<Matrix>(nPD4));
  tmLSM->set_positions(list, std::make_shared<Matrix>(nPD3), -1);
  assert(tmLSM == std::make_shared<Matrix>(nPD4));

  int triu0[2][3] = {{1, 2, 3}, {0, 5, 6}};
  assert(m1->triu(0) == std::make_shared<Matrix>(triu0));
  int triu1[2][3] = {{0, 2, 3}, {0, 0, 6}};
  assert(m1->triu(1) == std::make_shared<Matrix>(triu1));
  int triu2[2][3] = {{0, 0, 3}, {0, 0, 0}};
  assert(m1->triu(2) == std::make_shared<Matrix>(triu2));
  int triu3[2][3] = {{0, 0, 0}, {0, 0, 0}};
  assert(m1->triu(3) == std::make_shared<Matrix>(triu3));
  int triu1_2[3][3] = {{0, 2, 3}, {0, 0, 6}, {0, 0, 0}};
  assert(std::make_shared<Matrix>(nPD1)->triu(1) ==
         std::make_shared<Matrix>(triu1_2));

  int eye1[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
  assert(eye(3, 4) == std::make_shared<Matrix>(eye1));
  int eye2[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
  assert(eye(4, 3) == std::make_shared<Matrix>(eye2));
  int eye3[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  assert(eye(3, 3) == std::make_shared<Matrix>(eye3));

  int listColMatrix[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  int listColResult[3][1] = {{2}, {6}, {10}};
  assert(std::make_shared<Matrix>(listColMatrix)
             ->list_elems_by_column_position(2) ==
         std::make_shared<Matrix>(listColResult));

  double leetM[3][3] = {{1.337, 1.337, 1.337}, {1.337, 1.337, 1.337}, {1.337, 1.337, 1.337}};
  assert(std::make_shared<Matrix>(3, 3, 1.337) == std::make_shared<Matrix>(leetM));

  double mulTest[2][2] = {{1, 2}, {3, 4}};
  assert(std::make_shared<Matrix>(mulTest)->multiplyAllValues() == 24);
  assert(std::make_shared<Matrix>(mulTest)->sumAllValue() == 10);

  double listA[1][4] = {{1, 2, 3, 4}};
  std::vector<int> vecA = {1, 2, 3, 4};
  assert(std::make_shared<Matrix>(listA)->asVector<int>() == vecA);

  double lgTest[4][4] = {
      {2.25, 6.25, 111.25, 0.125},
      {0.125, 0.125, 0.125, 0.125},
      {494.25, 36, 45, 0.125},
      {0.125, 0.125, 0.125, 0.125}};
  double lgR[4][4] = {
      {lgamma(2.25), lgamma(6.25), lgamma(111.25), lgamma(0.125)},
      {lgamma(9881.25), lgamma(0.125), lgamma(95.125), lgamma(0.125)},
      {lgamma(494.25), lgamma(36), lgamma(45), lgamma(0.125)},
      {lgamma(0.125), lgamma(0.125), lgamma(0.125), lgamma(0.125)}};
  assert((std::make_shared<Matrix>(lgTest)->lgammaed()->inverse() * std::make_shared<Matrix>(lgR))->isIdentity());
  assert((std::make_shared<Matrix>(lgTest)->lgammaed() - std::make_shared<Matrix>(lgTest)->lgammaed())->sumAllValue() == 0);

#if ACCELERATE_MODE == ACCELERATE_MODE_NONE
  puts("Testing CPU ...");
#endif
#if ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
  puts("Testing OpenCL");
#endif
#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
  puts("Testing CUDA");
#endif

  for (int m = 1; m < 65536; m *= 2) {
    auto s = now(), e = now();
    printf("Prepping %dx100 test ...\n", m);
    auto
        m1 = sprand(m, 100, 1),
        m2 = sprand(m, 100, 1),
        m3 = sprand(m, 100, 1);
#if ACCELERATE_MODE == ACCELERATE_MODE_NONE
    m3 = m3->transpose();
#else
    puts("Testing move ...");
    s = now();
    m1->accelerate();
    e = now();
    auto ms1 = to_milliseconds(s, e);
    s = now();
    m2->accelerate();
    e = now();
    auto ms2 = to_milliseconds(s, e);
    s = now();
    m3->accelerate();
    e = now();
    auto ms3 = to_milliseconds(s, e);
    printf("Copy took %d, %d, %d ms.\n", ms1, ms2, ms3);

    m3 = m3->transpose();
    m3->accelerate();
#endif

    s = now();
    auto m4 = m2 * m3;
    m4->noop();
    e = now();
    auto ms4 = to_milliseconds(s, e);
    printf("Mult took %d ms.\n", ms4);

    puts("");
  }

  event_stop();
  return 0;
}
