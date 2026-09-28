CC = nvcc
CFLAGS = -ccbin g++ -gencode=arch=compute_80,code=sm_80 \
		 --extended-lambda \
	     -rdc=true -I$(JSON_ROOT)/include \
		 -Xcompiler "-Wreorder -Wall -O3"

SRCDIR = ../../10momentMHD2D ../../PIC2D ../../Interface2D ../../utils
SOURCES = $(shell find $(SRCDIR) -name '*.cu')
OBJS = $(SOURCES:.cu=.o)

MAIN = main.o

EXECUTABLE = program

.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(MAIN) $(MAINDEP) $(EXECUTABLE)
