extern "C" __global__ void daxpy(const double *x, double *y, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}