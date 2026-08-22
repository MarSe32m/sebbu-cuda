#!/bin/bash

nvcc daxpy_kernel.cu \
    --fatbin \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90 \
    -o kernels.fatbin

# nvcc -O3 -ptx -arch=compute_90 kernel.cu -o kernels.ptx