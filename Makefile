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

# Source directories
ASM_DIR = src/asm
C_DIR = src/c

# Assembly source files (order matters for dependencies)
ASM_SRCS = $(ASM_DIR)/mem.asm \
           $(ASM_DIR)/utils.asm \
           $(ASM_DIR)/error.asm \
           $(ASM_DIR)/simd.asm \
           $(ASM_DIR)/tensor.asm \
           $(ASM_DIR)/math_kernels.asm \
           $(ASM_DIR)/autograd.asm \
           $(ASM_DIR)/activations.asm \
           $(ASM_DIR)/nn_layers.asm \
           $(ASM_DIR)/losses.asm \
           $(ASM_DIR)/optimizers.asm \
           $(ASM_DIR)/dataset.asm \
           $(ASM_DIR)/model_io.asm \
           $(ASM_DIR)/config_parser.asm \
           $(ASM_DIR)/threads.asm \
           $(ASM_DIR)/training_ops.asm \
           $(ASM_DIR)/tests.asm \
           $(ASM_DIR)/verify.asm \
           $(ASM_DIR)/compat.asm \
           $(ASM_DIR)/main.asm

# C source files (complex features best written in C)
C_SRCS = $(C_DIR)/tb_logger.c \
         $(C_DIR)/pruning.c \
         $(C_DIR)/quantize.c \
         $(C_DIR)/batchnorm.c \
         $(C_DIR)/metrics_losses.c \
         $(C_DIR)/transforms.c \
         $(C_DIR)/embedding.c \
         $(C_DIR)/fuzzy.c \
         $(C_DIR)/conv2d.c \
         $(C_DIR)/pooling.c \
         $(C_DIR)/tensor_ops.c \
         $(C_DIR)/transformer.c \
         $(C_DIR)/activations_c.c \
         $(C_DIR)/optimizers_c.c \
         $(C_DIR)/rnn.c

# Library assembly sources (excluding main.asm for shared library)
LIB_ASM_SRCS = $(ASM_DIR)/mem.asm \
               $(ASM_DIR)/utils.asm \
               $(ASM_DIR)/error.asm \
               $(ASM_DIR)/simd.asm \
               $(ASM_DIR)/tensor.asm \
               $(ASM_DIR)/math_kernels.asm \
               $(ASM_DIR)/autograd.asm \
               $(ASM_DIR)/activations.asm \
               $(ASM_DIR)/nn_layers.asm \
               $(ASM_DIR)/losses.asm \
               $(ASM_DIR)/optimizers.asm \
               $(ASM_DIR)/dataset.asm \
               $(ASM_DIR)/model_io.asm \
               $(ASM_DIR)/config_parser.asm \
               $(ASM_DIR)/threads.asm \
               $(ASM_DIR)/training_ops.asm \
               $(ASM_DIR)/tests.asm \
               $(ASM_DIR)/verify.asm \
               $(ASM_DIR)/compat.asm \
               $(ASM_DIR)/neural_api.asm

# Library C sources (same as C_SRCS — all go into shared lib)
LIB_C_SRCS = $(C_SRCS)

# Object files — main binary
ASM_OBJS = $(patsubst $(ASM_DIR)/%.asm,%.o,$(ASM_SRCS))
C_OBJS = $(patsubst $(C_DIR)/%.c,%.o,$(C_SRCS))
OBJS = $(ASM_OBJS) $(C_OBJS)

# Object files — shared library (PIC)
LIB_ASM_PIC = $(patsubst $(ASM_DIR)/%.asm,%.pic.o,$(LIB_ASM_SRCS))
LIB_C_PIC = $(patsubst $(C_DIR)/%.c,%.pic.o,$(LIB_C_SRCS))
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
%.o: $(ASM_DIR)/%.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

# Assemble PIC objects for shared library
%.pic.o: $(ASM_DIR)/%.asm
	$(NASM) $(NASMFLAGS_PIC) -o $@ $<

# Compile C source files (non-PIC for main binary)
%.o: $(C_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile C source files (PIC for shared library)
%.pic.o: $(C_DIR)/%.c
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
mem.o: $(ASM_DIR)/mem.asm
utils.o: $(ASM_DIR)/utils.asm
tensor.o: $(ASM_DIR)/tensor.asm
math_kernels.o: $(ASM_DIR)/math_kernels.asm
autograd.o: $(ASM_DIR)/autograd.asm
activations.o: $(ASM_DIR)/activations.asm
nn_layers.o: $(ASM_DIR)/nn_layers.asm
losses.o: $(ASM_DIR)/losses.asm
optimizers.o: $(ASM_DIR)/optimizers.asm
dataset.o: $(ASM_DIR)/dataset.asm
model_io.o: $(ASM_DIR)/model_io.asm
config_parser.o: $(ASM_DIR)/config_parser.asm
main.o: $(ASM_DIR)/main.asm
