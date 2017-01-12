#include <cmath>
#include <random>

#ifndef sparfun_HPP
#define sparfun_HPP

SMatrix sprandsym(int n, double density) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> pos(0, n - 1);
  std::normal_distribution<double> distribution;
  SMatrix M = std::make_shared<Matrix>(n, n);
  for (int count = 0; count < std::round(n * (n + 1) * std::min(density, 1.0));
       ++count) {
    int r = pos(generator), c = pos(generator);
    M->data[_matrix_index_for(n, c, r)] = M->data[_matrix_index_for(n, r, c)] =
        distribution(generator);
  }
  return M;
}

SMatrix sprand(int m, int n, double density) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  auto nnzwanted = std::round(m * n * std::min(density, 1.0));
  SMatrix i = std::make_shared<Matrix>(m, n);
  for (auto count = 0; count < nnzwanted; ++count) {
    int r = std::floor(distribution(generator) * m);
    int c = std::floor(distribution(generator) * n);
    i->data[_matrix_index_for(n, r, c)] = distribution(generator);
  }
  return i;
}

#endif
