NVCC := nvcc
CXXBIN := /usr/bin/g++
TARGET := blocks_intro
SRC := blocks_intro.cu
NVCCFLAGS := -ccbin $(CXXBIN) -O2 -lineinfo

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
