#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
template <class T>
void getLaunchConfiguration(T t, int n, int *blocks, int *threads);
void vec_lgamma(double *a, double *c, const unsigned int n);
#endif
