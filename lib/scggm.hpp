#include "scggm_initialize.hpp"
#include "scggm_options.hpp"
#include "scggm_refit_step.hpp"
#include "scggm_sparse_step.hpp"
#include "scggm_zero_index.hpp"
#include <cstdio>
#include <memory>

struct scggm_return {
  bool error = false;
  scggm_theta Theta;
  SMatrix intercept;
};

scggm_return scggm(std::shared_ptr<Matrix> x, std::shared_ptr<Matrix> y,
                   double lambda_1, double lambda_2, scggm_options &options) {
  scggm_return OPT;
  int N = x->rows;
  if (N != y->rows) {
    OPT.error = true;
    puts("ERR: sample size inconsistent");
    return OPT;
  }

  scggm_theta Theta0;
  if (options.theta0 == nullptr) {
    if (options.verbose)
      puts("Generating theta0 ... ");
    Theta0 = scggm_initialize(x->cols, y->cols);
  } else {
    Theta0 = *options.theta0.get();
  }

  SMatrix cx, cy;
  if (options.centered_input) {
    cx = x;
    cy = y;
  } else {
    if (options.verbose)
      puts("Centering input ...");
    cy = y - y->mean()->repeat(N, 1);
    cx = x - x->mean()->repeat(N, 1);
  }

  if (options.verbose)
    puts("Calculating sparse step ...");
  auto stepRes = scggm_sparse_step(lambda_1, lambda_2, cx, cy,
                                   options.max_iterations, options.tolerance,
                                   options.verbose, options.eta, Theta0);
  auto raw_Theta = stepRes.theta;
  scggm_theta Theta;
  if (options.ifrefit) {
    if (options.verbose)
      puts("Finding zero index ...");
    auto zero_theta = scggm_zero_index(raw_Theta);
    if (options.verbose) {
      printf("%d %d\n", zero_theta.xy->rows, zero_theta.yy->rows);
      puts("Refitting ...");
    }
    auto sro = scggm_refit_step(cx, cy, zero_theta, options.max_iterations,
                                options.tolerance, options.verbose, options.eta,
                                raw_Theta);
    Theta = sro.Theta;
  } else {
    Theta = raw_Theta;
  }

  OPT.Theta = Theta;
  OPT.intercept = y->mean() + x->mean() * (Theta.xy * Theta.yy->inverse());
  return OPT;
}
