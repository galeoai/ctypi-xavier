#define THREADS 256
#include <cstdint>

__global__ void ctypi_v3(uint16_t *im1,uint16_t *im2, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
};
