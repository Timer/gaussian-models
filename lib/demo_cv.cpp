#include <cstdio>
#include "matrix.hpp"
#include "scggm_cov_decompose.hpp"
#include "scggm_cv.hpp"
#include "scggm_indirect_SNP_decompose.hpp"
#include "scggm_indirect_SNP_overall.hpp"
#include "scggm_predict.hpp"
#include "shared.hpp"

int main(int argc, char *argv[]) {
  event_start();

  //--------------------------------------------------------------------------
  // Sample run of the sparse CGGM algorithm with cross-validation
  //--------------------------------------------------------------------------

  // sCGGM demo with cross-validation
  // specify the search grid of regularization parameters
  std::vector<double> lambda1_seq = {0.16, 0.08, 0.04, 0.02};
  std::vector<double> lambda2_seq = {0.16, 0.08, 0.04, 0.02};

  // performs kcv-fold cross validation, kcv must be >= 3
  int kcv = 5;

  // loading traing data and test data
  auto xtrain = load("./data/xtrain.txt", false), ytrain = load("./data/ytrain.txt", false);
  auto xtest = load("./data/xtest.txt", false), ytest = load("./data/ytest.txt", false);

  printf("sCGGM demo with %d-fold cross-validation...\n", kcv);

  scggm_options options;
  options.verbose = true;
  options.max_iterations = 500;
  auto opt = scggm_cv(xtrain, ytrain, kcv, lambda1_seq, lambda2_seq, options);

  // compute prediction errors
  auto e = scggm_predict(opt.Theta, opt.intercept, xtest, ytest);
  printf("sCGGM demo  completed, test set prediction error: %g\n", e);

  // perform inference
  auto Beta = scggm_indirect_SNP_overall(opt.Theta);

  // decomposition of overall indirect SNP perturbations
  // passed by the k-th gene
  int k = 2;  // k = 1 ... 30
  auto Beta_k = scggm_indirect_SNP_decompose(opt.Theta, k);

  // decomposition of gene-expression covariance
  auto Cov = scggm_cov_decompose(opt.Theta, xtrain, ytrain);

  //TODO:
  //if ~exist('results/demo_cv', 'dir')
  //  mkdir('results/demo_cv');
  //end

  opt.Theta.xy->save("./results/demo_cv/optimal_Theta_xy.txt");
  opt.Theta.yy->save("./results/demo_cv/optimal_Theta_yy.txt");
  opt.intercept->save("./results/demo_cv/optimal_intercept.txt");
  double la[1][2] = {{opt.lambdas[0], opt.lambdas[1]}};
  std::make_shared<Matrix>(la)->save("./results/demo_cv/optimal_lambdas.txt");
  Beta->save("./results/demo_cv/Beta.txt");
  Beta_k->save("./results/demo_cv/Beta_k.txt");
  Cov.Overall->save("./results/demo_cv/Cov_Overall.txt");
  Cov.Network_Induced->save("./results/demo_cv/Cov_Network_Induced.txt");
  Cov.SNP_Induced->save("./results/demo_cv/Cov_SNP_Induced.txt");

  event_stop();
  return 0;
}
