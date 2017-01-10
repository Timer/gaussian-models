#include "scggm_theta.hpp"
#include <memory>

std::shared_ptr<scggm_theta> scggm_initialize(int J, int K) {
  std::shared_ptr<scggm_theta> Theta0 = std::make_shared<scggm_theta>();
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
