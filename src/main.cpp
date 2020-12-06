#include "opencv2/opencv.hpp"
#include <stdio.h>
#include "ctypi_base.h"
#include "ctypi_kernel.h"
#include <cstdint>
#include <chrono>
#include "H5Cpp.h"

using namespace cv;

int main(int argc, char *argv[])
{
    // read images
    uint16_t *im1;
    Mat img1 = imread("../tmp/raw.tif", IMREAD_ANYDEPTH);
    im1 = img1.ptr<uint16_t>(0);
     
    uint16_t *im2;
    Mat img2 = imread("../tmp/raw.tif", IMREAD_ANYDEPTH);
    im2 = img2.ptr<uint16_t>(0);

    // gen output image
    uint16_t *out;
    Mat m_out = Mat::zeros(img1.rows,img1.cols,img1.type());
    out = m_out.ptr<uint16_t>(0);

    //NUC read nuc files
    H5::H5File fid = H5::H5File("../tmp/nuc_tables.h5",H5F_ACC_RDONLY);
    H5::DataSet dataset = fid.openDataSet("offset");
    H5::DataSpace dataspace  = dataset.getSpace();

    float offset[2048*2048];
    dataset.read(offset, H5::PredType::NATIVE_FLOAT, dataspace);
     
    float dx=0,dy=0;
    int Height = img1.cols;
    int Width = img1.rows;
    printf("image size: (%d, %d)\n", Height, Width);
    auto start = std::chrono::high_resolution_clock::now();
    ctypi_v3_base(dx,dy,im1,im2,Height,Width);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    printf("cpu: dx=%f, dy=%f ===> duration = %ld[ms] \n",dx, dy, duration.count()/1000);

    start = std::chrono::high_resolution_clock::now();
    GPUdiff(out, im1, im2, Width*Height);
    GPUfilter_x(out, im1, Width, Height);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("GPUdiff: ===> duration = %ld[ms] \n", duration.count()/1000);

    imwrite("../data/images/diff.tif", m_out);
    
    return 0;
}
