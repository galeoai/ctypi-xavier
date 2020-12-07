#define THREADS 512
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
    if (i<size) {
	out[i] = im1[i]-im2[i];
    };
};

void GPUdiff(uint16_t *out, uint16_t *im1,uint16_t *im2, int size){
    uint16_t *d_out = NULL;
    cudaHostRegister(out, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_out, (void *)out, 0);
    uint16_t *d_im1 = NULL;
    cudaHostRegister(im1, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_im1, (void *)im1, 0);
    uint16_t *d_im2 = NULL;
    cudaHostRegister(im2, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_im2, (void *)im2, 0);

    int threadsPerBlock = THREADS;
    int blocksPerGrid =(size + threadsPerBlock - 1) / threadsPerBlock;

    diff<<<blocksPerGrid,threadsPerBlock>>>(d_out,d_im1,d_im2,size);
    //printf("calling the kernel\n");
    //cudaDeviceSynchronize();
    //cudaError_t cudaerr = cudaDeviceSynchronize();
    //if (cudaerr != cudaSuccess)
    //    printf("kernel launch failed with error \"%s\".\n",
    //           cudaGetErrorString(cudaerr));
    
    //printf("done\n");
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

    int ii0 = threadIdx.x;
    // copy pixel value to shared memory
    __shared__ uint16_t s[THREADS + 2*KERNEL_RADIUS];
    s[KERNEL_RADIUS + ii0] = in[i0+imageH*i1];
    __syncthreads();

    #pragma unroll
    for (int j0 = -KERNEL_RADIUS; j0 < KERNEL_RADIUS; ++j0) {
	out[i0+imageH*i1] += s[KERNEL_RADIUS + ii0 + j0]*c1[KERNEL_RADIUS+j0];
    };

};


void GPUfilter_x(uint16_t *out, uint16_t *in,int imageW,int imageH){
    int size = imageW*imageH;
    uint16_t *d_out = NULL;
    cudaHostRegister(out, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_out, (void *)out, 0);
    uint16_t *d_in = NULL;
    cudaHostRegister(in, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_in, (void *)in, 0);

    //int threadsPerBlock = THREADS;
    //int blocksPerGrid =(size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 threadsPerBlock(512,1);
    dim3 numBlocks(imageW/threadsPerBlock.x, imageH/threadsPerBlock.y);
    
    filter_x<<<numBlocks,threadsPerBlock>>>(d_out,d_in,imageW,imageH);
    //printf("calling the kernel\n");
    //cudaDeviceSynchronize();
    //cudaError_t cudaerr = cudaDeviceSynchronize();
    //if (cudaerr != cudaSuccess)
    //    printf("kernel launch failed with error \"%s\".\n",
    //           cudaGetErrorString(cudaerr));
    
    //printf("done\n");
    // clean up
    cudaHostUnregister(d_out);
    cudaHostUnregister(d_in);
}

///////////////////////////////////////////////////////////////////////////////
//                                  filter_y                                 //
///////////////////////////////////////////////////////////////////////////////
__global__ void filter_y(uint16_t *out,
			 uint16_t *in,
			 int imageW,
			 int imageH)
{
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    int i1 = blockIdx.y*blockDim.y + threadIdx.y;

    int ii0 = threadIdx.x;
    int ii1 = threadIdx.y;
    // copy pixel value to shared memory
    __shared__ uint16_t s[32][16 + 2*KERNEL_RADIUS];
    s[ii0][ii1+KERNEL_RADIUS] = in[i0+imageW*i1];
    // lower
    if(ii1==0) s[ii0][0] = in[i0+imageW*max(i1-3,0)];
    if(ii1==1) s[ii0][1] = in[i0+imageW*max(i1-3,0)];
    if(ii1==2) s[ii0][2] = in[i0+imageW*max(i1-3,0)];
    // upper
    if(ii1==13) s[ii0][KERNEL_RADIUS+16+0] = in[i0+imageW*min(i1+3,imageH)];
    if(ii1==14) s[ii0][KERNEL_RADIUS+16+1] = in[i0+imageW*min(i1+3,imageH)];
    if(ii1==15) s[ii0][KERNEL_RADIUS+16+2] = in[i0+imageW*min(i1+3,imageH)];

    __syncthreads();

    #pragma unroll
    for (int j1 = -KERNEL_RADIUS; j1 < KERNEL_RADIUS; ++j1) {
	out[i0+imageW*i1] += s[ii0][ii1+KERNEL_RADIUS+j1] *
	    c1[KERNEL_RADIUS+j1];
    };
};


void GPUfilter_y(uint16_t *out, uint16_t *in, int imageW, int imageH){
    int size = imageW*imageH;
    uint16_t *d_out = NULL;
    cudaHostRegister(out, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_out, (void *)out, 0);
    uint16_t *d_in = NULL;
    cudaHostRegister(in, size*sizeof(uint16_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&d_in, (void *)in, 0);

    dim3 threadsPerBlock(32,16);
    dim3 numBlocks(imageW/threadsPerBlock.x, imageH/threadsPerBlock.y);
    
    filter_y<<<numBlocks,threadsPerBlock>>>(d_out,d_in,imageW,imageH);
    // clean up
    cudaHostUnregister(d_out);
    cudaHostUnregister(d_in);
};
