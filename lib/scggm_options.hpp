#include "scggm_theta.hpp"
#include <memory>

#ifndef SCGGM_OPTIONS
#define SCGGM_OPTIONS

struct scggm_options {
  int max_iterations = 1000;
  double tolerance = 1e-7;
  bool verbose = false;
  double eta = 1.5;
  bool centered_input = false;
  bool ifrefit = true;
  scggm_theta *theta = nullptr;
};

#endif
