#include "matrix.hpp"
#include "scggm_evaluate.hpp"
#include "scggm_theta.hpp"
#include <cstdio>
#include <memory>

#ifndef scggm_refit_step_HPP
#define scggm_refit_step_HPP

struct scggm_refit_obj {
  scggm_theta Theta;
  SMatrix obj;
};

scggm_refit_obj scggm_refit_step(SMatrix cx, SMatrix cy, scggm_theta &z_theta,
                                 int max_iterations, double tolerance,
                                 bool verbose, double eta,
                                 scggm_theta &Theta0) {
  scggm_refit_obj ret;
  auto Sx = cx->transpose() * cx;
  auto Sy = cy->transpose() * cy;
  auto Sxy = cx->transpose() * cy;
  auto N = cx->rows;

  auto theta = Theta0;
  auto bconv = 0;
  /*
  TODO
  theta.xy(z_theta.xy) = 0;
  theta.yy(z_theta.yy) = 0; % constrain the sparsity pattern of the variable
  */
  ret.obj = std::make_shared<Matrix>(max_iterations, 1);
  double L = 1.0;
  auto nobj = 10;

  auto thk_0 = 2.0 / 3.0;
  auto ls_maxiter = 300;

  scggm_evaluate_obj se = scggm_evaluate(theta, Sx, Sxy, Sy, N, 'n', verbose);
  auto obj1 = se.value;
  auto init_flag = se.error;
  if (init_flag && verbose) {
    puts("sCGGM: error! refit initial Theta_yy not positive definite!");
  }

  ret.obj->set_position(1, obj1);
  auto xk = theta;
  auto zk = theta;
  auto thk = thk_0;

  int iter;
  for (iter = 2; iter <= max_iterations; ++iter) {
    // thk = 2 * thk / (2 + thk);
    thk = (sqrt(pow(thk, 4) + 4 * pow(thk, 2)) - pow(thk, 2)) / 2.0;
    scggm_theta y;
    y.xy = xk.xy->scalar(1 - thk) + zk.xy->scalar(thk);
    y.yy = xk.yy->scalar(1 - thk) + zk.yy->scalar(thk);
    auto se = scggm_evaluate(y, Sx, Sxy, Sy, N, 'y', verbose);
    auto fyk = se.value;
    auto flagy = se.error;
    auto grady = se.grad;
    // compute the objective and gradient for y
    /*
    TODO
    grady.xy(z_theta.xy) = 0;
    grady.yy(z_theta.yy) = 0;
    */

    // line search
    auto ik = 0;
    double fxk1 = 0.0;
    for (;;) {
      // gradient descent
      scggm_theta zk_grady;
      zk_grady.xy = y.xy - grady.xy->scalar(1.0 / (thk * L));
      zk_grady.yy = y.yy - grady.yy->scalar(1.0 / (thk * L));
      scggm_theta zk1 = zk_grady;

      // gradient descent
      scggm_theta xk1;
      xk1.xy = y.xy - grady.xy->scalar(1.0 / L);
      xk1.yy = y.yy - grady.yy->scalar(1.0 / L);

      auto se1 = scggm_evaluate(xk1, Sx, Sxy, Sy, N, 'n', verbose);
      auto fxk1 = se1.value;
      auto flagxk1 = se1.error;
      auto flagzk1 = zk1.yy->cholesky().error;

      if (!flagzk1 && !flagy && !flagxk1) { // xk1, zk1, y all positive definite
        scggm_theta xk1_y;
        xk1_y.xy = xk1.xy - y.xy;
        xk1_y.yy = xk1.yy - y.yy;
        auto lfxk1_y = fyk +
                       (grady.xy->list_elems_by_position()->transpose() *
                        (xk1_y.xy->list_elems_by_position()))
                           ->value() +
                       (grady.yy->list_elems_by_position()->transpose() *
                        (xk1_y.yy->list_elems_by_position()))
                           ->value();
        scggm_theta diffxk1;
        diffxk1.xy = xk1.xy - y.xy;
        diffxk1.yy = xk1.yy - y.yy;
        auto RHS =
            lfxk1_y +
            L / 2.0 *
                ((diffxk1.xy->list_elems_by_position()->power(2))->sumValue() +
                 (diffxk1.yy->list_elems_by_position()->power(2))->sumValue());
        auto LHS = fxk1;
        if (LHS <= RHS + tolerance) {
          xk = xk1;
          zk = zk1;
          bconv = 1;
          break; // line search converged
        }
      }

      ++ik;
      if (ik > ls_maxiter) {
        if (verbose) {
          printf("sCGGM: refit line search not converging,ik =%d\n", ik);
        }
        bconv = 0;
        iter = std::max(1, iter - 1);
        ret.Theta = xk;
        break;
      }

      L = L * eta;
    }
    ret.obj->set_position(iter, fxk1);
    if (bconv == 0) {
      break;
    }
    if (iter > nobj + 1) {
      double value = ret.obj->get_position(iter);
      double prevVals = ret.obj->get_position(iter - nobj);
      double avgimprovement = std::abs(prevVals - value) / nobj;
      double relAvgImpr = avgimprovement / std::abs(value);
      if (relAvgImpr < tolerance) {
        bconv = 1;
        break;
      }
    }
  }

  ret.Theta = xk;
  ret.obj = ret.obj->row_elems_by_position(1, iter);
  return ret;
}

#endif
