#include "matrix.hpp"
#include "scggm_theta.hpp"

#ifndef scggm_indirect_SNP_decompose_HPP
#define scggm_indirect_SNP_decompose_HPP

SMatrix scggm_indirect_SNP_decompose(scggm_theta Theta, int k) {
  auto iThetayy = Theta.yy->inverse();
  return Theta.xy->list_elems_by_column_position(k)->scalar(-1) *
         iThetayy->list_elems_by_column_position(k)->transpose();
}

#endif
