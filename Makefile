# define build parameters
DEBUG ?= 0
GPU   ?= 0

# define compiler suite
CXX  = g++

ifeq ($(GPU), 1)
NVCC = nvcc
else
NVCC = g++
endif

# define library paths
CUDADIR     ?= /usr/local/cuda
MAGMADIR    ?= ../magma-2.2.0
OPENBLASDIR ?= ../OpenBLAS-0.2.19

# define compiler flags, libraries
LIBS      = -lm
CXXFLAGS  = -std=c++11 \
            -I$(OPENBLASDIR)/include
NVCCFLAGS = -std=c++11 \
            -I$(CUDADIR)/include \
            -I$(MAGMADIR)/include \
            -I$(OPENBLASDIR)/include

ifeq ($(DEBUG), 1)
CXXFLAGS  += -g -pg -Wall
NVCCFLAGS += -g -pg -Xcompiler -Wall
else
CXXFLAGS  += -O3
NVCCFLAGS += -O3
endif

ifeq ($(GPU), 1)
LIBS      += -L$(MAGMADIR)/lib -lmagma \
             -L$(CUDADIR)/lib64 -lcudart -lcublas \
             -L$(OPENBLASDIR)/lib -lopenblas
else
LIBS      += -L$(OPENBLASDIR)/lib -lopenblas
NVCCFLAGS = $(CXXFLAGS)
endif

# define binary targets
BINS = face-rec test-image test-matrix

all: echo $(BINS)

echo:
	$(info DEBUG     = $(DEBUG))
	$(info GPU       = $(GPU))
	$(info CXX       = $(CXX))
	$(info NVCC      = $(NVCC))
	$(info LIBS      = $(LIBS))
	$(info CXXFLAGS  = $(CXXFLAGS))
	$(info NVCCFLAGS = $(NVCCFLAGS))

%.o: src/%.cpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

face-rec: bayes.o dataset.o ica.o identity.o image.o knn.o lda.o logger.o main.o math_utils.o matrix.o matrix_utils.o model.o pca.o timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-image: image.o logger.o math_utils.o matrix.o test_image.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-matrix: logger.o math_utils.o matrix.o test_matrix.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(BINS)
