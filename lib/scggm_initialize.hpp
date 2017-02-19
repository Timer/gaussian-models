#include "matrix.hpp"
#include "scggm_theta.hpp"
#include "sparfun.hpp"
#include <memory>

#ifndef scggm_initialize_HPP
#define scggm_initialize_HPP

scggm_theta scggm_initialize(int J, int K) {
  scggm_theta Theta0;
  do {
    SMatrix Ai;
    if (K <= 100) {
      Ai = sprandsym(K, 0.01);
      Theta0.xy = sprand(J, K, 0.01);
    } else {
      Ai = sprandsym(K, 0.001);
      Theta0.xy = sprand(J, K, 0.001);
    }
    Theta0.yy = Ai->scalar(0.01) * Ai->transpose() + eye(K, K)->scalar(0.7);
  } while (Theta0.yy->cholesky().error);
  return Theta0;
}

#endif
