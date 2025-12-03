# Neural Assembly Framework
# Minimal Deep Learning Framework in x86-64 Assembly
# Makefile for Linux, NASM, ELF64

# Assembler and flags
NASM = nasm
NASMFLAGS = -f elf64 -g -F dwarf

# Linker
LD = ld

# Output binary
TARGET = neural_framework

# Source files (order matters for dependencies)
SRCS = mem.asm \
       utils.asm \
       tensor.asm \
       math_kernels.asm \
       autograd.asm \
       activations.asm \
       nn_layers.asm \
       losses.asm \
       optimizers.asm \
       dataset.asm \
       model_io.asm \
       config_parser.asm \
       compat.asm \
       main.asm

# Object files
OBJS = $(SRCS:.asm=.o)

# Phony targets
.PHONY: all clean debug help run-test

# Default target
all: $(TARGET)

# Link all object files into final executable using gcc (links libc/libm)
CC = gcc
LDFLAGS = -lm -no-pie

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Assemble each source file
%.o: %.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

# Debug build with extra symbols
debug: NASMFLAGS += -g -F dwarf -l $*.lst
debug: clean all

# Clean build artifacts
clean:
	rm -f *.o *.lst $(TARGET)

# Run a quick test
run-test: $(TARGET)
	./$(TARGET) test

# Show help
help:
	@echo "Neural Assembly Framework Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build the framework (default)"
	@echo "  debug    - Build with debug symbols and listings"
	@echo "  clean    - Remove build artifacts"
	@echo "  run-test - Build and run gradient tests"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Usage after building:"
	@echo "  ./neural_framework train config.ini [model.bin]"
	@echo "  ./neural_framework infer config.ini model.bin"
	@echo "  ./neural_framework test"

# Dependencies
mem.o: mem.asm
utils.o: utils.asm
tensor.o: tensor.asm
math_kernels.o: math_kernels.asm
autograd.o: autograd.asm
activations.o: activations.asm
nn_layers.o: nn_layers.asm
losses.o: losses.asm
optimizers.o: optimizers.asm
dataset.o: dataset.asm
model_io.o: model_io.asm
config_parser.o: config_parser.asm
main.o: main.asm 
