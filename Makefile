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
            -I$(OPENBLASDIR)/include \
            -Wno-deprecated-gpu-targets

ifeq ($(DEBUG), 1)
CXXFLAGS  += -g -pg -Wall
NVCCFLAGS += -g -pg -Xcompiler -Wall
else
CXXFLAGS  += -O3 -Wno-unused-result
NVCCFLAGS += -O3 -Xcompiler -Wno-unused-result
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
OBJDIR = obj
OBJS = $(addprefix $(OBJDIR)/, \
	bayes.o \
	dataset.o \
	ica.o \
	identity.o \
	image.o \
	knn.o \
	lda.o \
	logger.o \
	main.o \
	math_utils.o \
	matrix.o \
	matrix_utils.o \
	model.o \
	pca.o \
	timer.o )

BINS = \
	face-rec \
	test-image \
	test-matrix

all: echo $(BINS)

echo:
	$(info DEBUG     = $(DEBUG))
	$(info GPU       = $(GPU))
	$(info CXX       = $(CXX))
	$(info NVCC      = $(NVCC))
	$(info LIBS      = $(LIBS))
	$(info CXXFLAGS  = $(CXXFLAGS))
	$(info NVCCFLAGS = $(NVCCFLAGS))

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: src/%.cpp src/%.h | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: src/%.cpp | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

face-rec: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-image: $(addprefix $(OBJDIR)/, image.o logger.o math_utils.o matrix.o test_image.o)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-matrix: $(addprefix $(OBJDIR)/, logger.o math_utils.o matrix.o test_matrix.o)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf $(OBJDIR) $(BINS)
