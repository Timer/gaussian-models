#include "matrix.hpp"
#include "scggm_theta.hpp"

#ifndef scggm_indirect_SNP_overall_HPP
#define scggm_indirect_SNP_overall_HPP

SMatrix scggm_indirect_SNP_overall(scggm_theta Theta) {
  return Theta.xy->scalar(-1) * Theta.yy->inverse();
}

#endif
