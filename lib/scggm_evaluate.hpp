#include "matrix.hpp"
#include "scggm_theta.hpp"
#include <cstdio>
#include <limits>

#ifndef scggm_evaluate_HPP
#define scggm_evaluate_HPP

struct scggm_evaluate_obj {
  double value = 0.0;
  bool error = false;
  scggm_theta grad;
};

scggm_evaluate_obj scggm_evaluate(scggm_theta theta, SMatrix Sx, SMatrix Sxy,
                                  SMatrix Sy, int N, char gradient,
                                  bool verbose) {
  scggm_evaluate_obj ret;
  Cholesky ch = theta.yy->cholesky();
  auto cyy = ch.matrix;
  if (ch.error) {
    if (verbose) {
      puts("sCGGM: Theta_yy not positive definite!");
    }
    ret.error = true;
    ret.value = std::numeric_limits<double>::infinity();
    ret.grad = theta;
    return ret;
  }

  double logdetyy = 2.0 * cyy->diag()->log()->sumValue();
  if (std::isnan(logdetyy) || std::isinf(logdetyy)) {
    if (verbose) {
      puts("sCGGM: logdet Theta_yy is Nan or Inf!");
    }
    ret.error = true;
    ret.value = std::numeric_limits<double>::infinity();
    ret.grad = theta;
    return ret;
  }

  // icyy	 = cyy \ eye(size(cyy,2));
  auto icyy = cyy->inverse();
  auto ithetayy = icyy * icyy->transpose();
  auto txyityy = theta.xy * ithetayy;
  auto XtXth = Sx * txyityy;
  auto txyXtXth = theta.xy->transpose() * Sx * txyityy;

  auto l1 = (theta.yy * Sy)->trace();
  auto l2 = (Sxy * theta.xy->transpose())->trace();
  auto l3 = txyXtXth->trace();
  ret.value = 0.5 * l1 + l2 + 0.5 * l3 - 0.5 * N * logdetyy;
  ret.value /= (double)N;

  if (gradient == 'y') {
    ret.grad.xy = (Sxy + XtXth)->scalar(1.0 / N);
    ret.grad.yy =
        (Sy - ithetayy->scalar(N) - ithetayy * txyXtXth)->scalar(0.5 / N);
  } else {
    // ret.grad = [];
  }
  return ret;
}

#endif
