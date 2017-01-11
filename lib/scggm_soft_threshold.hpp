#include "scggm_theta.hpp"

#ifndef scggm_soft_threshold_HPP
#define scggm_soft_threshold_HPP

scggm_theta scggm_soft_threshold(scggm_theta theta, double c1, double c2) {
  scggm_theta B;
  SMatrix Bxy = std::make_shared<Matrix>(theta.xy->rows, theta.xy->cols);
  /*
  TODO
  pos_idx=theta.xy(:)>c1;
  neg_idx=theta.xy(:)<-c1;

  Bxy(pos_idx)=theta.xy(pos_idx)-c1;
  Bxy(neg_idx)=theta.xy(neg_idx)+c1;
  */
  SMatrix Byy = theta.yy->diag()->diag();
  /*
  TODO
  uyy = triu( theta.yy,1 );
  pos_idx=uyy(:)>c2;
  neg_idx=uyy(:)<-c2;

  Byy(pos_idx) = uyy(pos_idx)-c2;
  Byy(neg_idx) = uyy(neg_idx)+c2;
  */
  Byy = Byy + Byy->transpose() - theta.yy->diag()->diag();
  B.xy = Bxy;
  B.yy = Byy;
  return B;
}

#endif
