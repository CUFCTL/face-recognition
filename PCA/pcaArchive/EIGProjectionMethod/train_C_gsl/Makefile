all: clean totalprogram

totalprogram: training.o ppm.o grayscale.o
	gcc -g -O3 grayscale.o training.o ppm.o -o training -L/usr/local/cula/lib64 -L/home/richard/ATLAS_BUILD/lib -lpthread -lcula_core -lcula_lapack -lcublas -lcudart -lgsl -lptcblas -latlas -lm

grayscale.o: grayscale.c
	gcc -g -O3 -c grayscale.c -I/usr/local/cula/include -I/usr/local/cuda/include

training.o: training.c
	gcc -g -O3 -c training.c -I/usr/local/cula/include -I/usr/local/cuda/include -I/home/richard/GSL/gsl-1.9

ppm.o: ppm.c
	gcc -g -O3 -c ppm.c -I/usr/local/cula/include -I/usr/local/cuda/include

clean:
	rm -rf *o training
