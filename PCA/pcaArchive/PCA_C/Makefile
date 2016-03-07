#   This make file will compile and link the necessary source code that 
#   will implement the PCA algorithm in C.
#
#   Necessary files for compilation are
#       ppm.c
#       pca.c
#       pca_host.c
#       matrix_ops.c
#       grayscale.c

all: clean totalprogram

totalprogram: grayscale.o matrix_ops.o pca_host.o pca.o ppm.o
	gcc -g -O3 grayscale.o matrix_ops.o pca_host.o pca.o ppm.o -o pca

grayscale.o: grayscale.c
	gcc -g -O3 -c grayscale.c

matrix_ops.o: matrix_ops.c
	gcc -g -O3 -c matrix_ops.c

pca_host.o: pca_host.c
	gcc -g -O3 -c pca_host.c

pca.o: pca.c
	gcc -g -O3 -c pca.c

ppm.o: ppm.c
	gcc -g -O3 -c ppm.c

clean:
	rm -rf *.o pca
