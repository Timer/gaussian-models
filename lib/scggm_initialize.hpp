#include "scggm_theta.hpp"
#include <memory>

#ifndef scggm_initialize_HPP
#define scggm_initialize_HPP

scggm_theta scggm_initialize(int J, int K) {
  scggm_theta Theta0;
  assert(false); // Not yet implemented!
  //TODO
  if (K <= 100) {
    // Ai = sprandsym(K,0.01);
    // Theta0.xy = sprand( J, K, 0.01 );
  } else {
    // Ai = sprandsym(K,0.001);
    // Theta0.xy = sprand( J, K, 0.001 );
  }
  // Theta0.yy = 0.01*Ai*Ai' + 0.7*speye(K,K);
  return Theta0;
}

#endif
