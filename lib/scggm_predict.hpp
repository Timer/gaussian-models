#ifndef SCGGM_PREDICT_HPP
#define SCGGM_PREDICT_HPP

#include "scggm_indirect_SNP_overall.hpp"

//--------------------------------------------------------------------------
// predict Y (gene-expression) based on X (genotype) and compute
// prediction error when true Y (gene-expression) is given.
//--------------------------------------------------------------------------
double scggm_predict(scggm_theta &Theta, SMatrix intercept, SMatrix x_ts, SMatrix y_ts) {
  auto K = Theta.yy->cols;
  auto N_ts = x_ts->rows;
  auto Beta = scggm_indirect_SNP_overall(Theta);

  auto Ey_ts = x_ts * Beta + intercept->repeat(N_ts, 1);
  if (y_ts->rows != N_ts) {
    throw "sCGGM:error! Genotype and expression test data sample size inconsistent!";
  }
  auto res = y_ts - Ey_ts;
  return res->power(2)->sumAllValue() / K / N_ts;
}

#endif
