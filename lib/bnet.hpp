#include "matrix.hpp"

#ifndef BNET_HPP
#define BNET_HPP

struct CPD;
typedef std::shared_ptr<CPD> SCPD;

struct CPD {
  SMatrix sizes, dirichlet, cpt;
};

#endif
