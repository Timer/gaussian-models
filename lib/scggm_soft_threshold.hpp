#include "scggm_theta.hpp"

#ifndef scggm_soft_threshold_HPP
#define scggm_soft_threshold_HPP

scggm_theta scggm_soft_threshold(scggm_theta theta, double c1, double c2) {
  scggm_theta B;
  /*
      Bxy=zeros(size(theta.xy));
      pos_idx=theta.xy(:)>c1;
      neg_idx=theta.xy(:)<-c1;

      Bxy(pos_idx)=theta.xy(pos_idx)-c1;
      Bxy(neg_idx)=theta.xy(neg_idx)+c1;

      Byy = diag(diag(theta.yy));
      uyy = triu( theta.yy,1 );
      pos_idx=uyy(:)>c2;
      neg_idx=uyy(:)<-c2;

      Byy(pos_idx) = uyy(pos_idx)-c2;
      Byy(neg_idx) = uyy(neg_idx)+c2;
      Byy = Byy + Byy'-diag(diag(theta.yy));

      B.xy = (Bxy);
      B.yy = (Byy);
  */
  return B;
}

#endif
