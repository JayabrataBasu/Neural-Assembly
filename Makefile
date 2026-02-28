# Neural Assembly Framework
# Minimal Deep Learning Framework in x86-64 Assembly
# Makefile for Linux, NASM, ELF64

# Assembler and flags
NASM = nasm
NASMFLAGS = -f elf64 -g -F dwarf
NASMFLAGS_PIC = -f elf64 -g -F dwarf -DPIC

# C compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -std=c11
CFLAGS_PIC = $(CFLAGS) -fPIC

# Output binary
TARGET = neural_framework
SHARED_LIB = libneural.so

# Python executable (prefer project virtualenv when present)
PYTHON ?= python3
ifneq ("$(wildcard .neuasm/bin/python)","")
PYTHON := ./.neuasm/bin/python
endif

# Assembly source files (order matters for dependencies)
ASM_SRCS = mem.asm \
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
           training_ops.asm \
           tests.asm \
           verify.asm \
           compat.asm \
           main.asm

# C source files (complex features best written in C)
C_SRCS = tb_logger.c \
         pruning.c \
         quantize.c \
         batchnorm.c \
         metrics_losses.c

# Library assembly sources (excluding main.asm for shared library)
LIB_ASM_SRCS = mem.asm \
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
               training_ops.asm \
               tests.asm \
               verify.asm \
               compat.asm \
               neural_api.asm

# Library C sources (same as C_SRCS — all go into shared lib)
LIB_C_SRCS = $(C_SRCS)

# Object files — main binary
ASM_OBJS = $(ASM_SRCS:.asm=.o)
C_OBJS = $(C_SRCS:.c=.o)
OBJS = $(ASM_OBJS) $(C_OBJS)

# Object files — shared library (PIC)
LIB_ASM_PIC = $(LIB_ASM_SRCS:.asm=.pic.o)
LIB_C_PIC = $(LIB_C_SRCS:.c=.pic.o)
PIC_OBJS = $(LIB_ASM_PIC) $(LIB_C_PIC)

# Phony targets
.PHONY: all clean debug help run-test lib shared install validate validate-smoke

# Default target
all: $(TARGET)

# Build shared library
lib: $(SHARED_LIB)
shared: $(SHARED_LIB)

# Link all object files into final executable using gcc (links libc/libm)
LDFLAGS = -lm -lpthread -no-pie
LDFLAGS_SHARED = -shared -lm -lpthread -Wl,-Bsymbolic

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Build shared library from PIC objects
$(SHARED_LIB): $(PIC_OBJS)
	$(CC) $(LDFLAGS_SHARED) -fPIC -o $@ $^

# Assemble each source file (non-PIC for main binary)
%.o: %.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

# Assemble PIC objects for shared library
%.pic.o: %.asm
	$(NASM) $(NASMFLAGS_PIC) -o $@ $<

# Compile C source files (non-PIC for main binary)
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile C source files (PIC for shared library)
%.pic.o: %.c
	$(CC) $(CFLAGS_PIC) -c -o $@ $<

# Debug build with extra symbols
debug: NASMFLAGS += -g -F dwarf -l $*.lst
debug: clean all

# Clean build artifacts
clean:
	rm -f *.o *.pic.o *.lst $(TARGET) $(SHARED_LIB)

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
	$(PYTHON) tools/verify_simple.py

# Run full verification (requires numpy)
verify-full: $(TARGET)
	$(PYTHON) tools/verify_correctness.py

# Run staged validation suite (build + datasets + tests + compatibility checks)
validate-smoke: $(TARGET) $(SHARED_LIB)
	$(PYTHON) tools/run_validation_suite.py --tier smoke

validate: $(TARGET) $(SHARED_LIB)
	$(PYTHON) tools/run_validation_suite.py --tier regression

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
