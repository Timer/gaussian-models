#include <cstdio>
#include <vector>
#include "matrix.hpp"
#include "rand.hpp"
#include "scggm_initialize.hpp"
#include "scggm_options.hpp"
#include "scggm_predict.hpp"
#include "scggm_refit_step.hpp"
#include "scggm_sparse_step.hpp"
#include "scggm_theta.hpp"
#include "scggm_zero_index.hpp"

struct scggm_return_cv {
  bool error = false;
  scggm_theta Theta;
  SMatrix intercept;
  double lambdas[2] = {0, 0};
};

scggm_return_cv scggm_cv(std::shared_ptr<Matrix> x, std::shared_ptr<Matrix> y, int kcv,
                         std::vector<double> lambda1_seq,
                         std::vector<double> lambda2_seq, scggm_options &options) {
  //default_lambdaseq = [0.32, 0.16, 0.08, 0.04, 0.02, 0.01];
  scggm_return_cv OPT;
  scggm_theta Theta0;
  if (options.theta0 == nullptr) {
    if (options.verbose)
      puts("Generating theta0 ... ");
    Theta0 = scggm_initialize(x->cols, y->cols);
  } else {
    Theta0 = *options.theta0.get();
  }

  int N0 = x->rows;
  if (y->rows != N0) {
    puts("sCGGM:error! Input data sample size inconsistent!");
    OPT.error = true;
    return OPT;
  }
  if (kcv < 3 || kcv >= N0) {
    printf("sCGGM:error! Cross validation cannot be %d fold\n", kcv);
    OPT.error = true;
    return OPT;
  }
  SMatrix x0, y0;
  if (options.centered_input) {
    x0 = x;
    y0 = y;
  } else {
    if (options.verbose) {
      puts("Centering input ...");
    }
    y0 = y - y->mean()->repeat(N0, 1);
    x0 = x - x->mean()->repeat(N0, 1);
  }

  int J = x->cols, K = y->cols;
  auto cv_indices = crossvalind(N0, kcv);
  auto cverr = std::make_shared<Matrix>(lambda1_seq.size(), lambda2_seq.size());
  auto minerr = 1e99;

  if (options.verbose) printf("J = %d, K = %d, sample size = %d\n", J, K, N0);

  double opt_lambda1 = -1, opt_lambda2 = -1;
  for (int i = 0; i < lambda1_seq.size(); ++i) {
    for (int j = 0; j < lambda2_seq.size(); ++j) {
      const auto lambda1 = lambda1_seq[i], lambda2 = lambda2_seq[j];
      for (int ff = 0; ff < kcv; ++ff) {
        auto trainData = findIndexByValue(cv_indices, ff, false);
        auto cvData = findIndexByValue(cv_indices, ff, true);
        // extract centered and uncentered training data
        auto cxtr = x0->extract_list_index(trainData, 0, x0->cols),
             cytr = y0->extract_list_index(trainData, 0, y0->cols),
             xtr = x->extract_list_index(trainData, 0, x->cols),
             ytr = y->extract_list_index(trainData, 0, y->cols);

        // extract uncentered cross-validation data
        auto xcv = x->extract_list_index(cvData, 0, x->cols),
             ycv = y->extract_list_index(cvData, 0, y->cols);

        if (options.verbose)
          puts("Calculating sparse step ...");
        // estimate a sparse estimate of Theta_xy and Theta_yy
        auto stepRes = scggm_sparse_step(lambda1, lambda2, cxtr, cytr,
                                         options.max_iterations, options.tolerance,
                                         options.verbose, options.eta, Theta0);
        auto raw_Theta = stepRes.theta;
        scggm_theta Theta;
        if (options.ifrefit) {
          if (options.verbose) {
            puts("Finding zero index ...");
          }
          auto zero_theta = scggm_zero_index(raw_Theta);
          if (options.verbose) {
            printf("Refitting (%d %d)\n", zero_theta.xy->rows, zero_theta.yy->rows);
          }
          auto sro = scggm_refit_step(cxtr, cytr, zero_theta, options.max_iterations,
                                      options.tolerance, options.verbose, options.eta,
                                      raw_Theta);
          Theta = sro.Theta;
        } else {
          Theta = raw_Theta;
        }
        // compute cross-validation error
        auto intercept = ytr->mean() + xtr->mean() * (Theta.xy * Theta.yy->inverse());
        auto e = scggm_predict(Theta, intercept, xcv, ycv);
        cverr->set_position(i + 1, j + 1, cverr->get_position(i + 1, j + 1) + e);
      }
      cverr->set_position(i + 1, j + 1, cverr->get_position(i + 1, j + 1) / (double) kcv);
      if (cverr->get_position(i + 1, j + 1) < minerr) {
        double minerr = cverr->get_position(i + 1, j + 1);
        opt_lambda1 = lambda1;
        opt_lambda2 = lambda2;
      }
      if (options.verbose) {
        printf("lambda_1 = %.2f\t lambda_2 = %.2f\t cross validation error = %.3f\n", lambda1, lambda2, cverr->get_position(i + 1, j + 1));
      }
    }
  }
  if (options.verbose) {
    printf("\ntraining sCGGM with optimal regularization parameters: \n");
    printf("optimal lambda_1 = %.2f\t optimal lambda_2 = %.2f... \n", opt_lambda1, opt_lambda2);
  }

  auto stepRes = scggm_sparse_step(opt_lambda1, opt_lambda2, x0, y0,
                                   options.max_iterations, options.tolerance,
                                   options.verbose, options.eta, Theta0);
  auto raw_Theta = stepRes.theta;
  scggm_theta Theta;
  if (options.ifrefit) {
    if (options.verbose) {
      puts("Finding zero index ...");
    }
    auto zero_theta = scggm_zero_index(raw_Theta);
    if (options.verbose) {
      printf("Refitting (%d %d)\n", zero_theta.xy->rows, zero_theta.yy->rows);
    }
    auto sro = scggm_refit_step(x0, y0, zero_theta, options.max_iterations,
                                options.tolerance, options.verbose, options.eta,
                                raw_Theta);
    Theta = sro.Theta;
  } else {
    Theta = raw_Theta;
  }

  OPT.intercept = y->mean() + x->mean() * (Theta.xy * Theta.yy->inverse());
  OPT.Theta = std::move(Theta);
  OPT.lambdas[0] = opt_lambda1;
  OPT.lambdas[1] = opt_lambda2;
  return OPT;
}
