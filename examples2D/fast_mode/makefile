CC = nvcc
CFLAGS = -O3 -arch=sm_89 -rdc=true --extended-lambda -Xcompiler "/wd 4819" -MD
SRCDIRS = ../../PIC2D_gpu ../../IdealMHD2D_gpu ../../Interface2D
SOURCES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.cu))
OBJS = $(patsubst %.cu,%.o,$(SOURCES))
DEPS = $(patsubst %.cu,%.d,$(SOURCES))
MAIN = main_fast_mode.cu
EXECUTABLE = program.exe
ADITIONALFILES = program.exp program.lib

.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN:.cu=.o)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN:.cu=.o) -o $(EXECUTABLE)

-include $(DEPS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN:.cu=.o): $(MAIN)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -rf $(OBJS) $(MAIN:.cu=.o) $(MAIN:.cu=.d) $(EXECUTABLE) $(ADITIONALFILES) $(DEPS)

clean_main:
	rm -rf $(MAIN:.cu=.o) $(MAIN:.cu=.d) $(EXECUTABLE) $(ADITIONALFILES)