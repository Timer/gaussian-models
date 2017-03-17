#ifndef scggm_cov_decompose_HPP
#define scggm_cov_decompose_HPP

#include "matrix.hpp"
#include "scggm_theta.hpp"

struct Cov {
  SMatrix Overall, Network_Induced, SNP_Induced;
};

//--------------------------------------------------------------------------
// compute covariance decomposition for gene-expression given an sCGGM,
// SNP data and gene-expression data.
//--------------------------------------------------------------------------
Cov scggm_cov_decompose(scggm_theta &Theta, SMatrix x, SMatrix y, bool centered_input) {
  Cov cov;
  auto iThetayy = Theta.yy->inverse();

  auto N = x->rows;
  if (y->rows != N) {
    puts("sCGGM:error! Genotype and expression data sample size inconsistent!");
    return cov;
  }

  if (!centered_input) {
    x = x - x->mean()->repeat(N, 1);
    y = y - y->mean()->repeat(N, 1);
  }
  cov.Overall = y->transpose() * y;
  cov.Network_Induced = iThetayy->scalar(N);
  cov.SNP_Induced = iThetayy * Theta.xy->transpose() * (x->transpose() * x) * Theta.xy * iThetayy;
  return cov;
}

Cov scggm_cov_decompose(scggm_theta &Theta, SMatrix x, SMatrix y) {
  return scggm_cov_decompose(Theta, x, y, false);
}
#endif
