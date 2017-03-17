#include <assert.h>
#include <stdlib.h>

#ifndef RAND_HPP
#define RAND_HPP

unsigned int rand_inclusive(unsigned int min, unsigned int max) {
  assert(max >= min);
  unsigned int
      delta = 1 + max - min,
      b = RAND_MAX / delta, c = b * delta;
  int r;
  do {
    r = rand();
  } while (r >= c);
  return min + r / b;
}

void shuffle_int(int c, int *a) {
  int b, d;
  while (c) b = rand_inclusive(0, --c), d = a[c], a[c] = a[b], a[b] = d;
}

std::vector<int> crossvalind(int size, int k) {
  int *indices = new int[size];
  for (int i = 0; i < size; ++i) {
    indices[i] = i % k;
  }
  shuffle_int(size, indices);
  std::vector<int> v;
  for (int i = 0; i < size; ++i) {
    v.push_back(indices[i]);
  }
  return std::move(v);
}

template <class T>
std::vector<int> findIndexByValue(std::vector<T> &v, T value, bool match) {
  std::vector<int> o;
  for (int i = 0; i < v.size(); ++i) {
    if ((v[i] == value) ^ match) {
      o.push_back(i);
    }
  }
  return std::move(o);
}

#endif
