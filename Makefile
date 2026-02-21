# Neural Assembly Framework
# Minimal Deep Learning Framework in x86-64 Assembly
# Makefile for Linux, NASM, ELF64

# Assembler and flags
NASM = nasm
NASMFLAGS = -f elf64 -g -F dwarf
NASMFLAGS_PIC = -f elf64 -g -F dwarf -DPIC

# Linker
LD = ld

# Output binary
TARGET = neural_framework
SHARED_LIB = libneural.so

# Source files (order matters for dependencies)
SRCS = mem.asm \
       utils.asm \
       error.asm \
       simd.asm \
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
       threads.asm \
       tests.asm \
       verify.asm \
       compat.asm \
       main.asm

# Library source files (excluding main.asm for shared library)
LIB_SRCS = mem.asm \
           utils.asm \
           error.asm \
           simd.asm \
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
           threads.asm \
           tests.asm \
           verify.asm \
           compat.asm \
           neural_api.asm

# Object files
OBJS = $(SRCS:.asm=.o)
LIB_OBJS = $(LIB_SRCS:.asm=.o)
PIC_OBJS = $(LIB_SRCS:.asm=.pic.o)

# Phony targets
.PHONY: all clean debug help run-test lib shared install validate validate-smoke

# Default target
all: $(TARGET)

# Build shared library
lib: $(SHARED_LIB)
shared: $(SHARED_LIB)

# Link all object files into final executable using gcc (links libc/libm)
CC = gcc
LDFLAGS = -lm -lpthread -no-pie
LDFLAGS_SHARED = -shared -lm -lpthread -Wl,-Bsymbolic

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Build shared library from PIC objects
$(SHARED_LIB): $(PIC_OBJS)
	$(CC) $(LDFLAGS_SHARED) -fPIC -o $@ $^

# Assemble each source file
%.o: %.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

# Assemble PIC objects for shared library
%.pic.o: %.asm
	$(NASM) $(NASMFLAGS_PIC) -o $@ $<

# Debug build with extra symbols
debug: NASMFLAGS += -g -F dwarf -l $*.lst
debug: clean all

# Clean build artifacts
clean:
	rm -f *.o *.lst $(TARGET) $(SHARED_LIB)

# Install shared library (requires sudo)
install: $(SHARED_LIB)
	install -d /usr/local/lib
	install -m 755 $(SHARED_LIB) /usr/local/lib/
	install -d /usr/local/include
	install -m 644 neural_api.h /usr/local/include/
	ldconfig

# Run a quick test
run-test: $(TARGET)
	./$(TARGET) test

# Run mathematical verification
verify: $(TARGET)
	python3 tools/verify_simple.py

# Run full verification (requires numpy)
verify-full: $(TARGET)
	python3 tools/verify_correctness.py

# Run staged validation suite (build + datasets + tests + compatibility checks)
validate-smoke: $(TARGET) $(SHARED_LIB)
	python3 tools/run_validation_suite.py --tier smoke

validate: $(TARGET) $(SHARED_LIB)
	python3 tools/run_validation_suite.py --tier regression

# Show help
help:
	@echo "Neural Assembly Framework Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the framework executable (default)"
	@echo "  lib/shared - Build shared library (libneural.so)"
	@echo "  debug      - Build with debug symbols and listings"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install shared library (requires sudo)"
	@echo "  run-test   - Build and run unit tests"
	@echo "  verify     - Build and run mathematical verification"
	@echo "  verify-full - Build and run full verification (requires numpy)"
	@echo "  validate-smoke - Run smoke validation suite"
	@echo "  validate   - Run regression validation suite"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Usage after building:"
	@echo "  ./neural_framework train config.ini [model.bin]"
	@echo "  ./neural_framework infer config.ini model.bin"
	@echo "  ./neural_framework test"
	@echo ""
	@echo "Python bindings:"
	@echo "  make lib && python3 -c 'import pyneural'"

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
