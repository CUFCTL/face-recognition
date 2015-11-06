# makefile for testing matrix functions
# Version: 1
#
# use make -f makefilename to run other makefile
# -lm is used to link in the math library
#
# -Wall turns on all warning messages 
#
#
comp = gcc
comp_flags = -g -Wall 
comp_libs = -lm -llapack -lblas 
#-llapacke
#-lgfortran
#comp = cc

testMatrixOps : testMatrixOps.o
	$(comp) $(comp_flags) testMatrixOps.o -o testMatrixOps $(comp_libs)

testMatrixOps.o: testMatrixOps.c matrix_manipulation.h
	$(comp) $(comp_flags) -c testMatrixOps.c
	
clean :
	rm -f *.o testMatrixOps core

