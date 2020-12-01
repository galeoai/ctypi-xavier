#define THREADS 256
#include <cstdint>
#include <stdio.h>

#include "ctypi_kernel.h"

#define KERNEL_RADIUS 3
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c1[KERNEL_LENGTH] = {
    0.0116850998497429921230139626686650444753468036651611328125,
    -0.0279730819380002923568717676516826031729578971862792968750,
    0.2239007887600356350166208585505955852568149566650390625000,
    0.5847743866564433234955799889576155692338943481445312500000,
    0.2239007887600356350166208585505955852568149566650390625000,
    -0.0279730819380002923568717676516826031729578971862792968750,
    0.0116850998497429921230139626686650444753468036651611328125 };

///////////////////////////////////////////////////////////////////////////////
//                                    diff                                   //
///////////////////////////////////////////////////////////////////////////////
// out = im1-im2 
__global__ void diff(uint16_t *out, uint16_t *im1,uint16_t *im2, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<size) out[i] = im1[i]-im2[i];
    printf("i = %d\n", i);
};

void GPUdiff(uint16_t *out, uint16_t *im1,uint16_t *im2, int size){
    uint16_t *d_out = NULL;
    cudaHostRegister(out, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_out, (void *)out, 0);
    uint16_t *d_im1 = NULL;
    cudaHostRegister(im1, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&im1, (void *)im1, 0);
    uint16_t *d_im2 = NULL;
    cudaHostRegister(im2, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&im2, (void *)im2, 0);

    int threadsPerBlock = THREADS;
    int blocksPerGrid =(size + threadsPerBlock - 1) / threadsPerBlock;

    //diff<<<blocksPerGrid,threadsPerBlock>>>(d_out,d_im1,d_im2,size);
    diff<<<10,1>>>(d_out,d_im1,d_im2,size);
    
    // clean up
    cudaHostUnregister(d_out);
    cudaHostUnregister(d_im1);
    cudaHostUnregister(d_im2);
}

///////////////////////////////////////////////////////////////////////////////
//                                  filter_x                                 //
///////////////////////////////////////////////////////////////////////////////
__global__ void filter_x(uint16_t *out,
			 uint16_t *in,
			 int imageW,
			 int imageH)
{
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    int i1 = blockIdx.y*blockDim.y + threadIdx.y;
    
    //for (int j0 = -KERNEL_RADIUS; j0 < KERNEL_RADIUS; ++j0) {
    // 	if( ((i0+j0)=>0) && ((i0+j0)<imageW)) {
    // 	    out[i0+imageH*i1]+=in[i0+imageH*i1+j0]*c1[KERNEL_RADIUS+j0];
    // 	};
    //};
};


