#if ACCELERATE_MODE == ACCELERATE_MODE_CUDA
template <class T>
void getLaunchConfiguration(T t, int n, int *blocks, int *threads);
double *cu_lgammed(const int rows, const int cols, double *iData);
double *cu_add(const int rows, const int cols, double *m1, double *m2);
double *cu_sub(const int rows, const int cols, double *m1, double *m2);
#endif
