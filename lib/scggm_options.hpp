#include <memory>
#include "scggm_theta.hpp"

#ifndef SCGGM_OPTIONS
#define SCGGM_OPTIONS

struct scggm_options {
  int max_iterations = 1000;
  double tolerance = 1e-7;
  bool verbose = true;
  double eta = 1.5;
  bool centered_input = false;
  bool ifrefit = true;
  std::shared_ptr<scggm_theta> theta0 = nullptr;
};

#endif
