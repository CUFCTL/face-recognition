# determine build (debug, release)
ifndef BUILD
BUILD = debug
endif

# determine matrix library (netlib, mkl, cuda)
ifndef MATLIB
MATLIB = netlib
endif

# determine compiler suite (gcc/g++, icc/icpc, nvcc)
ifeq ($(MATLIB), netlib)
CXX = g++
NVCC = g++
else ifeq ($(MATLIB), mkl)
CXX = icpc
NVCC = icpc
else ifeq ($(MATLIB), cuda)
NVCC = nvcc
endif

# determine compiler, library flags
LIBS = -lm
CXXFLAGS =
NVCCFLAGS = -x c++

ifeq ($(BUILD), release)
CXXFLAGS += -O3
else ifeq ($(BUILD), debug)
CXXFLAGS += -pg -Wall
endif

ifeq ($(MATLIB), netlib)
LIBS += -lblas -llapacke
NVCCFLAGS += $(CXXFLAGS)
else ifeq ($(MATLIB), mkl)
LIBS += -mkl
CXXFLAGS += -D INTEL_MKL
NVCCFLAGS += $(CXXFLAGS)
else ifeq ($(MATLIB), cuda)
LIBS += -lcudart -lcublas
endif

INCS = src/database.h src/image_entry.h src/image.h src/logger.h src/matrix.h src/timer.h
OBJS = database.o ica.o image_entry.o image.o lda.o main.o matrix.o pca.o test_image.o test_matrix.o timer.o
BINS = face-rec test-image test-matrix

all: config $(BINS)

config:
	$(info BUILD     = $(BUILD))
	$(info MATLIB    = $(MATLIB))
	$(info CXX       = $(CXX))
	$(info NVCC      = $(NVCC))
	$(info LIBS      = $(LIBS))
	$(info CXXFLAGS  = $(CXXFLAGS))
	$(info NVCCFLAGS = $(NVCCFLAGS))

database.o: src/database.cpp src/database.h image_entry.o image.o matrix.o timer.o
	$(CXX) -c $(CXXFLAGS) -o $@ $<

ica.o: src/ica.cpp src/database.h matrix.o timer.o
	$(CXX) -c $(CXXFLAGS) -o $@ $<

image_entry.o: src/image_entry.cpp src/image_entry.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

image.o: src/image.cpp src/image.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

lda.o: src/lda.cpp src/database.h matrix.o timer.o
	$(CXX) -c $(CXXFLAGS) -o $@ $<

main.o: src/main.cpp database.o timer.o
	$(CXX) -c $(CXXFLAGS) -o $@ $<

matrix.o: src/matrix.cu src/matrix.h image.o
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

pca.o: src/pca.cpp src/database.h matrix.o timer.o
	$(CXX) -c $(CXXFLAGS) -o $@ $<

timer.o: src/timer.cpp src/timer.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

test_image.o: src/test_image.cpp image.o matrix.o
	$(CXX) -c $(CXXFLAGS) -o $@ $<

test_matrix.o: src/test_matrix.cpp matrix.o
	$(CXX) -c $(CXXFLAGS) -o $@ $<

face-rec: database.o ica.o image_entry.o image.o lda.o main.o matrix.o pca.o timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-image: image.o matrix.o test_image.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test-matrix: matrix.o test_matrix.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(BINS)
	rm -rf test_images train_images
