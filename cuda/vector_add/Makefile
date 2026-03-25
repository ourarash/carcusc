NVCC := nvcc
CXXBIN := /usr/bin/g++
TARGET := vector_add
SRC := vector_add.cu
NVCCFLAGS := -ccbin $(CXXBIN) -O2 -lineinfo

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
