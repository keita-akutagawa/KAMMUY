CC = nvcc
CFLAGS = -O3 -arch=sm_89 -rdc=true --extended-lambda -Xcompiler "/wd 4819"
SRCDIRS = ../../lib_pic2D_gpu_single ../../IdealMHD2D_gpu ../../Interface2D
SOURCES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.cu))
OBJS = $(patsubst %.cu,%.o,$(SOURCES))
MAIN = main_sample.cu
EXECUTABLE = program.exe


.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN:.cu=.o)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN:.cu=.o) -o $(EXECUTABLE)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN:.cu=.o): $(MAIN)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(MAIN:.cu=.o) $(EXECUTABLE)