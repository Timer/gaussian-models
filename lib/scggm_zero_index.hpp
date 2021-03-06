#include "scggm_theta.hpp"

#ifndef scggm_zero_index_HPP
#define scggm_zero_index_HPP

scggm_theta scggm_zero_index(scggm_theta &theta, int eps) {
  scggm_theta nz_theta = scggm_theta();
  auto idsxy = theta.xy->abs()->find_positions(eps, false, true);
  auto idsyy = theta.yy->abs()->find_positions(eps, false, true);
  nz_theta.xy = idsxy;
  nz_theta.yy = idsyy;
  return nz_theta;
}

scggm_theta scggm_zero_index(scggm_theta &theta) {
  return scggm_zero_index(theta, 0);
}

#endif
