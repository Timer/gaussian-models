#include "matrix.hpp"
#include "scggm_evaluate.hpp"
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

scggm_sparse_obj scggm_sparse_step(int lambda1, int lambda2, SMatrix cx,
                                   SMatrix cy, int maxiter, double tol,
                                   bool verbose, double eta,
                                   scggm_theta theta0) {
  scggm_sparse_obj ret;
  auto transposedCx = cx->transpose();
  auto Sx = transposedCx * cx, Sy = cy->transpose() * cy,
       Sxy = transposedCx * cy;
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
  auto init_flag = er.flag;
  if (init_flag == 1 && verbose) {
    puts("sCGGM: error! initial Theta_yy not positive definite!");
  }
  // TODO: obj(1) = obj1 + scggm_penalty(theta, lambda1, lambda2);
  auto xk = theta, zk = theta;
  double thk = thk_0;
  for (int iter = 2; iter <= maxiter; ++iter) {
    thk = (sqrt(pow(thk, 4) + 4 * pow(thk, 2)) - pow(thk, 2)) / 2.0;
    scggm_theta y;
    y.xy = xk.xy->scalar(1 - thk) + zk.xy->scalar(thk);
    y.yy = xk.yy->scalar(1 - thk) + zk.yy->scalar(thk);
    auto er2 = scggm_evaluate(y, Sx, Sxy, Sy, N, 'y', verbose);
    double fyk = er2.value;
    int flagy = er2.flag;
    auto grady = er2.grad;
    int ik = 0;
    while (true) {
      // gradient step
      scggm_theta zk_grady;
      zk_grady.xy = (zk.xy->subtract(1.0 / (L * thk))) * grady.xy;
      zk_grady.yy = (zk.yy->subtract(1.0 / (L * thk))) * grady.yy;
      // proximal step
      auto zk1 = scggm_soft_threshold(zk_grady, 2.0 * lambda1 / (thk * L),
                                      2.0 * lambda2 / (thk * L));
      // gradient step
      scggm_theta y_grady;
      y_grady.xy = y.xy - grady.xy->scalar(1.0 / L);
      y_grady.yy = y.yy - grady.yy->scalar(1.0 / L);
      // proximal step
      auto xk1 =
          scggm_soft_threshold(y_grady, 2.0 * lambda1 / L, 2.0 * lambda2 / L);
      auto er3 = scggm_evaluate(xk1, Sx, Sxy, Sy, N, 'n', verbose);
      auto fxk1 = er3.value;
      auto flagxk1 = er3.flag;
      // TODO:  [~, flagzk1]    = chol(zk1.yy);
      int flagzk1 = 0; // TODO: remove
      if (flagzk1 == 0 && flagy == 0 &&
          flagxk1 == 0) { // xk1, zk1, y all positive definite
        scggm_theta xk1_y;
        xk1_y.xy = xk1.xy - y.xy;
        xk1_y.yy = xk1.yy - y.yy;

        double lfxk1_y = fyk +
                         (grady.xy->list_elems_by_col()->transpose() *
                          xk1_y.xy->list_elems_by_col())
                             ->value() +
                         (grady.yy->list_elems_by_col()->transpose() *
                          xk1_y.yy->list_elems_by_col())
                             ->value();
        scggm_theta diffxk1y;
        diffxk1y.xy = xk1.xy - y.xy;
        diffxk1y.yy = xk1.yy - y.yy;
        double RHS =
            lfxk1_y +
            L / 2.0 *
                ((diffxk1y.xy->list_elems_by_col()->power(2))->sumValue() +
                 (diffxk1y.yy->list_elems_by_col()->power(2))->sumValue());
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
    // TODO: obj(iter)  = fxk1 + scggm_penalty( xk, lambda1, lambda2);
    if (bconv == 0) {
      break;
    }
    if (iter > nobj + 1) {
      /*
      value           = obj(iter);
      prevVals        = obj(iter - nobj);
      avgimprovement  = abs( prevVals - value )/nobj;
      relAvgImpr      = avgimprovement / abs( value ) ;
      */
      double relAvgImpr; // TODO: remove
      if (relAvgImpr < tol) {
        bconv = 1;
        break;
      }
    }
  }
  ret.theta = xk;
  /*
  ret.obj   = ret.obj(1:iter);
  */
  return ret;
}

#endif
