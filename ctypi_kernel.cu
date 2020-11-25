#define THREADS 256

__global__ void ctypi_v3(){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
}; 
