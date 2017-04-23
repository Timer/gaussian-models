#include <cstdio>
#include "scggm.hpp"
#include "scggm_cov_decompose.hpp"
#include "scggm_indirect_SNP_decompose.hpp"
#include "scggm_indirect_SNP_overall.hpp"
#include "shared.hpp"

int main(int argc, char *argv[]) {
  event_start();

  //--------------------------------------------------------------------------
  // Sample run of the sparse CGGM algorithm without cross-validation
  //--------------------------------------------------------------------------
  // specify regularization parameters
  double lambda_1 = 0.1;
  double lambda_2 = 0.1;

  SMatrix xtrain = load("./data/xtrain.txt", false),
          ytrain = load("./data/ytrain.txt", false);

  printf(
      "sCGGM demo...\nJ = %d, K = %d, sample size = %d\nRegularization "
      "parameters: lambda_1 = %g, lambda_2 = %g\n",
      xtrain->cols, ytrain->cols, xtrain->rows, lambda_1, lambda_2);

  scggm_options options;
  auto opt = scggm(xtrain, ytrain, lambda_1, lambda_2, options);
  if (opt.error) {
    puts("SCGGM error -- terminating.");
    return 1;
  }

  // overall indirect SNP perturbations
  puts("Computing overall indirect SNP perturbations");
  auto Beta = scggm_indirect_SNP_overall(opt.Theta);

  // decomposition of overall indirect SNP perturbations
  // passed by the k-th gene
  int k = 2;
  printf(
      "Computing decomposition of overall indirect SNP perturbations on "
      "gene %d.\n",
      k);
  auto Beta_k = scggm_indirect_SNP_decompose(opt.Theta, k);

  // decomposition of gene-expression covariance
  auto Cov = scggm_cov_decompose(opt.Theta, xtrain, ytrain);

  //TODO:
  //if ~exist('results/demo', 'dir')
  //  mkdir('results/demo');
  //end

  opt.Theta.xy->save("./results/demo/optimal_Theta_xy.txt");
  opt.Theta.yy->save("./results/demo/optimal_Theta_yy.txt");
  opt.intercept->save("./results/demo/optimal_intercept.txt");
  Beta->save("./results/demo/Beta.txt");
  Beta_k->save("./results/demo/Beta_k.txt");
  Cov.Overall->save("./results/demo/Cov_Overall.txt");
  Cov.Network_Induced->save("./results/demo/Cov_Network_Induced.txt");
  Cov.SNP_Induced->save("./results/demo/Cov_SNP_Induced.txt");
  event_stop();
  return 0;
}
