#include "matrix.hpp"
#include "scggm_theta.hpp"
#include <cmath>

#ifndef scggm_penalty_HPP
#define scggm_penalty_HPP

double scggm_penalty(scggm_theta x, double lambda1, double lambda2) {
  return lambda1 * x.xy->list_elems_by_position()->abs()->sumValue() +
         lambda2 * x.yy->list_elems_by_position()->abs()->sumValue() -
         lambda2 * x.yy->diag()->abs()->sumValue();
}

#endif
