MATRIX_PATH = ../matrix

CC = gcc
CFLAGS = -I $(MATRIX_PATH) -g -Wall
LFLAGS = -L $(MATRIX_PATH)

OBJS = matrix.o
BINS = pca-train pca-recognize

%.o: $(MATRIX_PATH)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all: $(BINS)

pca-train: $(OBJS) pca_train.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -lm -llapacke pca_train.c -o $@

pca-recognize: $(OBJS) pca_recognize.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -lm -llapacke pca_recognize.c -o $@

clean:
	rm -f *.o *.dat *.ppm $(BINS)
