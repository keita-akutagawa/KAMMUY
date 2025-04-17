CC = nvcc
CFLAGS  = -ccbin mpic++ \
          -I$(AMGX_ROOT)/include \
          -L$(AMGX_ROOT)/lib \
          -gencode=arch=compute_60,code=sm_80 \
		  --extended-lambda \
          -rdc=true \
          -Xcompiler "-Wall -O3"
SRCDIRS = ../../PIC2D_multiGPU ../../IdealMHD2D_multiGPU ../../Interface2D_multiGPU
SOURCES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.cu))
OBJS = $(patsubst %.cu,%.o,$(SOURCES))
MAIN = main_fast_mode.o
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE) \
	-lamgxsh -lcusparse -lcublas -lcudart -ldl -lmpi \
		-Xlinker -rpath -Xlinker $(AMGX_ROOT)/lib

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_fast_mode.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(EXECUTABLE) $(MAIN)
