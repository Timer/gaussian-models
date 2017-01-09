#include "scggm_options.hpp"
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

  /*
  %% estimate a sparse estimate of Theta_xy and Theta_yy
  [raw_Theta] = scggm_sparse_step( lambda_1 , lambda_2,cx, cy,  maxiter, tol,
  verbose, eta, Theta0);

  if ifrefit
        %% refit the parameters
        zero_theta = scggm_zero_index(raw_Theta);
        Theta = scggm_refit_step(cx, cy, zero_theta, maxiter, tol, verbose, eta,
  raw_Theta);
  else
        Theta = raw_Theta;
  end

  %% record results
  OPT.Theta = Theta;
  OPT.intercept = mean(y) + mean(x) * (Theta.xy * inv(Theta.yy));
  */
}

int main(int argc, char *argv[]) { return 0; }
