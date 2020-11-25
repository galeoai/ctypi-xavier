CXX=g++
CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv4`

all:
	$(CXX) $(CFLAGS) -c main.cpp -o main.o
	nvcc $(CUDAFLAGS) -c ctypi_kernel.cu -o ctypi_kernel.o
	$(CXX)  main.o `pkg-config --libs opencv4 cuda-10.2` kernel.o -lcudart -o main

clean:
	rm -f *.o main

