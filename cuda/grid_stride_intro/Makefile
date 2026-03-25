NVCC := nvcc
CXXBIN := /usr/bin/g++
TARGET := grid_stride_intro
SRC := grid_stride_intro.cu
NVCCFLAGS := -ccbin $(CXXBIN) -O2 -lineinfo

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
