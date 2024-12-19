CC = nvcc
CFLAGS = -ccbin=mpic++ -gencode=arch=compute_60,code=sm_80 -rdc=true --extended-lambda -Xcompiler "-Wall -O3"
SRCDIRS = ../../PIC2D_multiGPU ../../IdealMHD2D_multiGPU ../../Interface2D_multiGPU
SOURCES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.cu))
OBJS = $(patsubst %.cu,%.o,$(SOURCES))
MAIN = main_shock_tube.o
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_shock_tube.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(EXECUTABLE)
