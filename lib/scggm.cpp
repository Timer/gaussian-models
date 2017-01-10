#include "scggm_options.hpp"
#include "scggm_refit_step.hpp"
#include "scggm_sparse_step.hpp"
#include "scggm_zero_index.hpp"
#include <iostream>
#include <memory>

void scggm(std::shared_ptr<Matrix> x, std::shared_ptr<Matrix> y, int lambda_1,
           int lambda_2, scggm_options options) {
  int N = x->rows;
  if (N != y->rows) {
    std::cout << "ERR: sample size inconsistent" << std::endl;
    return;
  }

  std::shared_ptr<Matrix> cx, cy;
  if (options.centered_input) {
    cx = x;
    cy = y;
  } else {
    cy = y - y->mean()->repeat(N, 1);
    cx = x - x->mean()->repeat(N, 1);
  }

  auto stepRes = scggm_sparse_step(
      lambda_1, lambda_2, cx, cy, options.max_iterations, options.tolerance,
      options.verbose, options.eta, *options.theta);
  auto raw_Theta = *stepRes.theta.get();
  scggm_theta Theta;
  if (options.ifrefit) {
    auto zero_theta = scggm_zero_index(raw_Theta);
    Theta = scggm_refit_step(cx, cy, zero_theta, options.max_iterations,
                             options.tolerance, options.verbose, options.eta,
                             raw_Theta);
  } else {
    Theta = raw_Theta;
  }
  // TODO
  /*
  % return
  OPT.Theta = Theta;
  OPT.intercept = mean(y) + mean(x) * (Theta.xy * inv(Theta.yy));
  */
}

int main(int argc, char *argv[]) { return 0; }
