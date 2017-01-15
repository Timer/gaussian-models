#include "scggm_theta.hpp"

#ifndef scggm_soft_threshold_HPP
#define scggm_soft_threshold_HPP

scggm_theta scggm_soft_threshold(scggm_theta theta, double c1, double c2,
                                 bool verbose) {
  scggm_theta B;
  SMatrix Bxy = std::make_shared<Matrix>(theta.xy->rows, theta.xy->cols);
  auto pos_idx =
      theta.xy->list_elems_by_position()->find_positions(c1, true, false);
  auto neg_idx =
      theta.xy->list_elems_by_position()->find_positions(-c1, false, false);
  Bxy->set_positions(pos_idx, theta.xy, -c1);
  Bxy->set_positions(neg_idx, theta.xy, c1);
  SMatrix Byy = theta.yy->diag()->diag();
  auto uyy = theta.yy->triu(1);
  pos_idx = uyy->list_elems_by_position()->find_positions(c2, true, false);
  neg_idx = uyy->list_elems_by_position()->find_positions(-c2, false, false);
  Byy->set_positions(pos_idx, uyy, -c2);
  Byy->set_positions(neg_idx, uyy, c2);

  Byy = Byy + Byy->transpose() - theta.yy->diag()->diag();
  // B.xy = (Bxy);
  // B.yy = (Byy);
  B.xy = Bxy;
  B.yy = Byy;
  return B;
}

#endif
