#include "nuc_kernel.h"
#include <stdio.h>

#define THREADS 512

__global__ void nuc(uint16_t *out,const float *gain,const float *offset, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<size)	out[i] = (uint16_t)(gain[i]*(out[i] - offset[i]));
}


void GPUnuc(uint16_t *out, float *gain, float *offset, int n){
    // d_offset zero-copy
    float *d_offset = NULL;
    cudaHostRegister(offset, n*sizeof(float), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_offset, (void *)offset, 0);
    // d_gain zero-copy
    float *d_gain = NULL;
    cudaHostRegister(gain, n*sizeof(float), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_gain, (void *)gain, 0);
    // d_out zero-copy
    uint16_t *d_out = NULL;
    cudaHostRegister(out, n*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_out, (void *)out, 0);

    int threadsPerBlock = THREADS;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    nuc<<<blocksPerGrid,threadsPerBlock>>>(d_out, d_gain, d_offset,n);

    // clean up
    cudaHostUnregister(d_out);
    cudaHostUnregister(d_offset);
    cudaHostUnregister(d_gain);

    return;
};
