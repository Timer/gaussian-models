#include "matrix.hpp"
#include "scggm_evaluate.hpp"
#include "scggm_penalty.hpp"
#include "scggm_soft_threshold.hpp"
#include "scggm_theta.hpp"
#include <cstdio>
#include <memory>

#ifndef SCGGM_SPARSE_STEP_HPP
#define SCGGM_SPARSE_STEP_HPP

struct scggm_sparse_obj {
  scggm_theta theta;
  SMatrix obj;
};

scggm_sparse_obj scggm_sparse_step(double lambda1, double lambda2, SMatrix cx,
                                   SMatrix cy, int maxiter, double tol,
                                   bool verbose, double eta,
                                   scggm_theta theta0) {
  scggm_sparse_obj ret;
  auto Sx = cx->multiply(cx, true, false), Sy = cy->multiply(cy, true, false),
       Sxy = cx->multiply(cy, true, false);
  auto N = cx->rows;
  int nobj = 10;
  int bconv = 0;
  ret.obj = std::make_shared<Matrix>(maxiter, 1);
  auto theta = theta0;
  double L = 1;
  double thk_0 = 2.0 / 3.0;
  int ls_maxiter = 300;
  scggm_evaluate_obj er = scggm_evaluate(theta, Sx, Sxy, Sy, N, 'n', verbose);
  auto obj1 = er.value;
  if (er.error && verbose) {
    puts("sCGGM: error! initial Theta_yy not positive definite!");
    throw 1;
  }
  ret.obj->set_position(1, obj1 + scggm_penalty(theta, lambda1, lambda2));
  auto xk = theta, zk = theta;
  double thk = thk_0;
  int iter;
  for (iter = 2; iter <= maxiter; ++iter) {
    thk = (sqrt(pow(thk, 4) + 4 * pow(thk, 2)) - pow(thk, 2)) / 2.0;
    scggm_theta y;
    y.xy = xk.xy->scalar(1 - thk) + zk.xy->scalar(thk);
    y.yy = xk.yy->scalar(1 - thk) + zk.yy->scalar(thk);
    auto er2 = scggm_evaluate(y, Sx, Sxy, Sy, N, 'y', verbose);
    double fyk = er2.value;
    auto grady = er2.grad;
    int ik = 0;
    double fxk1 = 0.0;
    while (true) {
      // gradient step
      scggm_theta zk_grady;
      zk_grady.xy = y.xy - grady.xy->scalar(1.0 / (thk * L));
      zk_grady.yy = y.yy - grady.yy->scalar(1.0 / (thk * L));
      // proximal step
      auto zk1 = scggm_soft_threshold(zk_grady, 2.0 * lambda1 / (thk * L),
                                      2.0 * lambda2 / (thk * L), verbose);
      // gradient step
      scggm_theta y_grady;
      y_grady.xy = y.xy - grady.xy->scalar(1.0 / L);
      y_grady.yy = y.yy - grady.yy->scalar(1.0 / L);
      // proximal step
      auto xk1 = scggm_soft_threshold(y_grady, 2.0 * lambda1 / L,
                                      2.0 * lambda2 / L, verbose);
      auto er3 = scggm_evaluate(xk1, Sx, Sxy, Sy, N, 'n', verbose);
      fxk1 = er3.value;
      if (!zk1.yy->cholesky().error && !er2.error &&
          !er3.error) { // xk1, zk1, y all positive definite
        scggm_theta xk1_y;
        xk1_y.xy = xk1.xy - y.xy;
        xk1_y.yy = xk1.yy - y.yy;

        double lfxk1_y = fyk +
                         (grady.xy->list_elems_by_position()->multiply(
                              xk1_y.xy->list_elems_by_position(), true, false))
                             ->value() +
                         (grady.yy->list_elems_by_position()->multiply(
                              xk1_y.yy->list_elems_by_position(), true, false))
                             ->value();
        scggm_theta diffxk1y;
        diffxk1y.xy = xk1.xy - y.xy;
        diffxk1y.yy = xk1.yy - y.yy;
        double RHS =
            lfxk1_y +
            L / 2.0 *
                ((diffxk1y.xy->list_elems_by_position()->power(2))->sumValue() +
                 (diffxk1y.yy->list_elems_by_position()->power(2))->sumValue());
        if (fxk1 <= RHS + tol) {
          xk = xk1;
          zk = zk1;
          bconv = 1;
          break; // line search converged
        }
      }
      ++ik;
      if (ik > ls_maxiter) {
        if (verbose) {
          printf("sCGGM: line search not converging,ik = %d\n", ik);
        }
        bconv = 0;
        iter = std::max(1, iter - 1);
        ret.theta = xk;
        break;
      }
      L = L * eta;
    }
    ret.obj->set_position(iter, fxk1 + scggm_penalty(xk, lambda1, lambda2));
    if (bconv == 0) {
      break;
    }
    if (iter > nobj + 1) {
      double value = ret.obj->get_position(iter);
      double prevVals = ret.obj->get_position(iter - nobj);
      double avgimprovement = std::abs(prevVals - value) / nobj;
      double relAvgImpr = avgimprovement / std::abs(value);
      if (relAvgImpr < tol) {
        bconv = 1;
        break;
      }
    }
  }
  ret.theta = xk;
  ret.obj = ret.obj->row_elems_by_position(1, iter);
  return ret;
}

#endif
