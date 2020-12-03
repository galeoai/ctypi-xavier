#ifndef CTYPI_KERNEL_H
#define CTYPI_KERNEL_H
#include <cstdint>

void GPUdiff(uint16_t *out, uint16_t *im1,uint16_t *im2, int size);
void GPUfilter_x(uint16_t *out, uint16_t *in,int imageW,int imageH);

#endif
