#include "matrix.hpp"
#include "scggm_theta.hpp"

#ifndef scggm_indirect_SNP_decompose_HPP
#define scggm_indirect_SNP_decompose_HPP

SMatrix scggm_indirect_SNP_decompose(scggm_theta Theta, int k) {
  auto iThetayy = Theta.yy->inverse();
  // TODO: Beta_k = - Theta.xy(:,k) * iThetayy(:,k)';
}

#endif
