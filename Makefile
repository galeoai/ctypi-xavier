CXX=g++
CFLAGS= -std=c++11 -Wall -I. `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4 cuda-10.2`
CUDAFLAGS = 
TARGET = stam

src = $(wildcard *.cpp)
obj = $(src:.cpp=.o)
src_cuda = $(wildcard *.cu)
obj_cuda = $(src_cuda:.cu=.o)
obj_all = $(obj) $(obj_cuda)

$(TARGET): $(obj_all)
	$(CXX) $(obj) $(LIBS) $(obj_cuda) -lcudart -o $@

main.o:main.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

cytpi_base.o:cytpi_base.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

ctypi_kernel.o: ctypi_kernel.cu
	nvcc -c $(CUDAFLAGS) $< -o $@

nuc_kernel.o: nuc_kernel.cu
	nvcc -c $(CUDAFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(obj_all) $(TARGET)

