#include "matrix.hpp"
#include <memory>

#ifndef SCGGM_THETA
#define SCGGM_THETA

class scggm_theta {
  std::shared_ptr<Matrix> xy = nullptr;
  std::shared_ptr<Matrix> yy = nullptr;
};

#endif
