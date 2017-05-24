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

obj:
	mkdir -p obj

obj/%.o: src/%.cpp | obj
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

face-rec: obj/bayes.o obj/dataset.o obj/ica.o obj/identity.o obj/image.o obj/knn.o obj/lda.o obj/logger.o obj/main.o obj/math_utils.o obj/matrix.o obj/matrix_utils.o obj/model.o obj/pca.o obj/timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-image: obj/image.o obj/logger.o obj/math_utils.o obj/matrix.o obj/test_image.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-matrix: obj/logger.o obj/math_utils.o obj/matrix.o obj/test_matrix.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf obj $(BINS)
