; =============================================================================
; neural_api.asm - C-Compatible API for Neural Assembly Framework
; =============================================================================
; This module provides exported C-callable functions with proper error handling
; and return codes for use with Python ctypes, C programs, and other FFI.
; =============================================================================

section .data
    align 8
    
    ; Version string
    version_str:        db "Neural Assembly Framework v1.0", 0
    
    ; Initialization flag
    initialized:        dd 0
    
    ; SIMD level names
    simd_name_scalar:   db "Scalar (no SIMD)", 0
    simd_name_sse2:     db "SSE2", 0
    simd_name_avx:      db "AVX", 0
    simd_name_avx2:     db "AVX2", 0
    simd_name_avx512:   db "AVX-512", 0
    
    ; Error code constants (matching neural_api.h)
    NEURAL_OK                   equ 0
    NEURAL_ERR_NULL_POINTER     equ 1
    NEURAL_ERR_OUT_OF_MEMORY    equ 2
    NEURAL_ERR_INVALID_ARGUMENT equ 3
    NEURAL_ERR_SHAPE_MISMATCH   equ 4
    NEURAL_ERR_DTYPE_MISMATCH   equ 5
    NEURAL_ERR_FILE_NOT_FOUND   equ 6
    NEURAL_ERR_FILE_READ        equ 7
    NEURAL_ERR_FILE_WRITE       equ 8
    NEURAL_ERR_PARSE_ERROR      equ 9
    NEURAL_ERR_INVALID_CONFIG   equ 10
    NEURAL_ERR_TENSOR_TOO_LARGE equ 11
    NEURAL_ERR_INVALID_DTYPE    equ 12
    NEURAL_ERR_DIM_MISMATCH     equ 13
    NEURAL_ERR_NOT_IMPLEMENTED  equ 14
    NEURAL_ERR_INTERNAL         equ 15

    ; Module struct offsets (from nn_layers.asm)
    MODULE_PARAMS       equ 8
    MODULE_FORWARD_FN   equ 24
    
    ; Tensor struct offsets
    TENSOR_DATA         equ 0
    TENSOR_SHAPE        equ 8
    TENSOR_STRIDE       equ 16
    TENSOR_NDIM         equ 24
    TENSOR_DTYPE        equ 32
    TENSOR_FLAGS        equ 36

section .bss
    align 8
    last_error:         resd 1

section .text

; External functions from other modules
extern mem_init
extern mem_alloc
extern mem_alloc_aligned
extern mem_free
extern mem_zero
extern mem_copy

extern tensor_create
extern tensor_free
extern tensor_zeros
extern tensor_numel
extern tensor_fill
extern tensor_copy
extern tensor_reshape
extern tensor_data_size

extern error_set
extern error_get
extern error_clear
extern error_get_message

extern ew_add
extern ew_sub
extern ew_mul
extern ew_div
extern matmul

extern relu_forward
extern sigmoid_forward
extern softmax_forward
extern tanh_forward
extern gelu_forward
extern leaky_relu_forward
extern elu_forward
extern selu_forward
extern swish_forward
extern mish_forward
extern hardswish_forward
extern softplus_forward
extern hardtanh_forward

; Note: linear_free, optimizer_step, dataset_size may not exist in all implementations
extern linear_create
extern linear_forward

extern mse_loss
extern cross_entropy_loss

extern sgd_create
extern adam_create
extern adamw_create
extern optimizer_step

; Sequential container functions
extern neural_sequential_create
extern neural_sequential_free
extern neural_sequential_add
extern neural_sequential_forward
extern neural_sequential_size
extern neural_sequential_get
extern neural_sequential_parameters
extern optimizer_free
extern clip_grad_norm_
extern clip_grad_value_

extern node_create
extern node_free
extern backward
extern node_relu
extern node_sigmoid
extern node_softmax

extern dataset_load_csv
extern dataset_free
extern dataset_get_batch
extern dataset_size

extern config_parse
extern config_free
extern config_get_int
extern config_get_float
extern config_get_string

extern detect_cpu_features

; Tensor struct offsets (must match tensor.asm)
%define TENSOR_DATA     0
%define TENSOR_NDIM     8
%define TENSOR_SHAPE    16
%define TENSOR_STRIDE   24
%define TENSOR_DTYPE    32
%define TENSOR_FLAGS    36

; Export API functions
global neural_init
global neural_shutdown
global neural_version
global neural_get_last_error
global neural_get_error_message
global neural_clear_error
global neural_tensor_create
global neural_tensor_zeros
global neural_tensor_ones
global neural_tensor_random
global neural_tensor_from_data
global neural_tensor_from_buffer
global neural_tensor_free
global neural_tensor_data
global neural_tensor_ndim
global neural_tensor_shape
global neural_tensor_stride
global neural_tensor_numel
global neural_tensor_dtype
global neural_tensor_bytes
global neural_tensor_fill
global neural_tensor_copy
global neural_tensor_reshape
global neural_tensor_is_contiguous
global neural_tensor_make_contiguous
global neural_buffer_info
global neural_add
global neural_sub
global neural_mul
global neural_div
global neural_matmul
global neural_sum
global neural_mean
global neural_relu
global neural_sigmoid
global neural_tanh
global neural_softmax
global neural_gelu
global neural_leaky_relu
global neural_elu
global neural_selu
global neural_swish
global neural_mish
global neural_hardswish
global neural_softplus
global neural_hardtanh
global neural_clip_grad_norm
global neural_clip_grad_value
global neural_linear_create
global neural_linear_free
global neural_linear_forward
global neural_linear_weight
global neural_linear_bias
global neural_mse_loss
global neural_cross_entropy_loss
global neural_sgd_create
global neural_adam_create
global neural_adamw_create
global neural_optimizer_free
global neural_optimizer_step
global neural_optimizer_zero_grad
global neural_node_create
global neural_node_free
global neural_backward
global neural_node_grad
global neural_dataset_load_csv
global neural_dataset_free
global neural_dataset_size
global neural_dataset_get_batch
global neural_config_load
global neural_config_free
global neural_config_get_int
global neural_config_get_float
global neural_config_get_string
global neural_get_simd_level
global neural_get_simd_name

; =============================================================================
; neural_init - Initialize the framework
; Returns: int (0 on success, error code on failure)
; =============================================================================
neural_init:
    push rbp
    mov rbp, rsp
    
    ; Check if already initialized
    mov eax, [rel initialized]
    test eax, eax
    jnz .already_init
    
    ; Initialize memory subsystem
    call mem_init
    test eax, eax
    jnz .mem_fail
    
    ; Detect SIMD capabilities
    call detect_cpu_features
    
    ; Mark as initialized
    mov dword [rel initialized], 1
    mov dword [rel last_error], NEURAL_OK
    
    xor eax, eax
    jmp .done

.already_init:
    xor eax, eax
    jmp .done

.mem_fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY
    mov eax, NEURAL_ERR_OUT_OF_MEMORY
    
.done:
    pop rbp
    ret

; =============================================================================
; neural_shutdown - Shutdown the framework
; =============================================================================
neural_shutdown:
    push rbp
    mov rbp, rsp
    
    mov dword [rel initialized], 0
    mov dword [rel last_error], NEURAL_OK
    
    pop rbp
    ret

; =============================================================================
; neural_version - Get version string
; Returns: const char*
; =============================================================================
neural_version:
    lea rax, [rel version_str]
    ret

; =============================================================================
; neural_get_last_error - Get last error code
; Returns: int
; =============================================================================
neural_get_last_error:
    mov eax, [rel last_error]
    ret

; =============================================================================
; neural_get_error_message - Get error message for code
; Arguments: int error_code
; Returns: const char*
; =============================================================================
neural_get_error_message:
    push rbp
    mov rbp, rsp
    
    call error_get_message
    
    pop rbp
    ret

; =============================================================================
; neural_clear_error - Clear last error
; =============================================================================
neural_clear_error:
    mov dword [rel last_error], NEURAL_OK
    call error_clear
    ret

; =============================================================================
; neural_tensor_create - Create a new tensor
; Arguments:
;   RDI = const uint64_t* shape
;   RSI = uint64_t ndim
;   EDX = int dtype
; Returns: NeuralTensor* (NULL on error)
; =============================================================================
neural_tensor_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    ; Save arguments
    mov r12, rdi            ; shape
    mov r13, rsi            ; ndim
    mov r14d, edx           ; dtype
    
    ; Validate arguments
    test rdi, rdi
    jz .null_shape
    
    test rsi, rsi
    jz .zero_ndim
    
    cmp edx, 1
    ja .invalid_dtype
    
    ; Call internal tensor_create
    ; tensor_create(ndim, shape, dtype)
    mov rdi, r13            ; ndim
    mov rsi, r12            ; shape
    mov edx, r14d           ; dtype
    call tensor_create
    
    test rax, rax
    jz .alloc_fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null_shape:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax
    jmp .done

.zero_ndim:
    ; Create scalar tensor
    mov rdi, 0
    xor esi, esi
    mov edx, r14d
    call tensor_create
    jmp .check_result

.invalid_dtype:
    mov dword [rel last_error], NEURAL_ERR_INVALID_DTYPE
    xor eax, eax
    jmp .done

.alloc_fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY
    xor eax, eax
    jmp .done

.check_result:
    test rax, rax
    jz .alloc_fail
    mov dword [rel last_error], NEURAL_OK

.done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_tensor_zeros - Create tensor filled with zeros
; Arguments: same as neural_tensor_create
; Returns: NeuralTensor*
; =============================================================================
neural_tensor_zeros:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 8
    
    ; Create tensor first
    call neural_tensor_create
    test rax, rax
    jz .done
    
    mov r12, rax
    
    ; Fill with zeros
    mov rdi, rax
    xorps xmm0, xmm0        ; value = 0.0
    call tensor_fill
    
    mov rax, r12
    
.done:
    add rsp, 8
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_tensor_ones - Create tensor filled with ones
; =============================================================================
neural_tensor_ones:
    push rbp
    mov rbp, rsp
    push r12
    sub rsp, 8
    
    ; Create tensor first
    call neural_tensor_create
    test rax, rax
    jz .done
    
    mov r12, rax
    
    ; Fill with ones
    mov rdi, rax
    mov eax, 0x3f800000     ; 1.0f in IEEE 754
    movd xmm0, eax
    call tensor_fill
    
    mov rax, r12
    
.done:
    add rsp, 8
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_tensor_free - Free a tensor
; Arguments: NeuralTensor* tensor
; =============================================================================
neural_tensor_free:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .done
    
    call tensor_free
    
.done:
    pop rbp
    ret

; =============================================================================
; neural_tensor_data - Get data pointer
; Arguments: NeuralTensor* tensor
; Returns: void*
; =============================================================================
neural_tensor_data:
    test rdi, rdi
    jz .null
    mov rax, [rdi + TENSOR_DATA]
    ret
.null:
    xor eax, eax
    ret

; =============================================================================
; neural_tensor_ndim - Get number of dimensions
; Arguments: NeuralTensor* tensor
; Returns: uint64_t
; =============================================================================
neural_tensor_ndim:
    test rdi, rdi
    jz .null
    mov rax, [rdi + TENSOR_NDIM]
    ret
.null:
    xor eax, eax
    ret

; =============================================================================
; neural_tensor_shape - Get shape pointer
; Arguments: NeuralTensor* tensor
; Returns: const uint64_t*
; =============================================================================
neural_tensor_shape:
    test rdi, rdi
    jz .null
    mov rax, [rdi + TENSOR_SHAPE]
    ret
.null:
    xor eax, eax
    ret

; =============================================================================
; neural_tensor_stride - Get stride pointer (in bytes)
; Arguments: NeuralTensor* tensor
; Returns: const int64_t*
; =============================================================================
neural_tensor_stride:
    test rdi, rdi
    jz .null
    mov rax, [rdi + TENSOR_STRIDE]
    ret
.null:
    xor eax, eax
    ret

; =============================================================================
; neural_tensor_numel - Get total number of elements
; Arguments: NeuralTensor* tensor
; Returns: uint64_t
; =============================================================================
neural_tensor_numel:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call tensor_numel
    jmp .done

.null:
    xor eax, eax

.done:
    pop rbp
    ret

; =============================================================================
; neural_tensor_dtype - Get data type
; Arguments: NeuralTensor* tensor
; Returns: int (-1 if null)
; =============================================================================
neural_tensor_dtype:
    test rdi, rdi
    jz .null
    mov eax, [rdi + TENSOR_DTYPE]
    ret
.null:
    mov eax, -1
    ret

; =============================================================================
; neural_tensor_bytes - Get size in bytes
; Arguments: NeuralTensor* tensor
; Returns: uint64_t
; =============================================================================
neural_tensor_bytes:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call tensor_data_size
    jmp .done

.null:
    xor eax, eax

.done:
    pop rbp
    ret

; =============================================================================
; neural_tensor_fill - Fill tensor with value
; Arguments:
;   RDI = NeuralTensor* tensor
;   XMM0 = double value
; Returns: int (error code)
; =============================================================================
neural_tensor_fill:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    ; Convert double to float if needed
    cvtsd2ss xmm0, xmm0
    call tensor_fill
    
    xor eax, eax
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop rbp
    ret

; =============================================================================
; neural_tensor_is_contiguous - Check if tensor is C-contiguous (row-major)
; Arguments: NeuralTensor* tensor
; Returns: int (1 if contiguous, 0 if not, -1 on error)
; =============================================================================
neural_tensor_is_contiguous:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    test rdi, rdi
    jz .is_contig_null
    
    mov r12, rdi                    ; tensor
    mov r13, [rdi + TENSOR_NDIM]    ; ndim
    
    ; Scalar or 0-d tensor is contiguous
    test r13, r13
    jz .is_contig_yes
    
    ; Get dtype size
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, 0
    je .is_contig_f32
    mov rbx, 8                      ; float64
    jmp .is_contig_check
.is_contig_f32:
    mov rbx, 4                      ; float32

.is_contig_check:
    ; Check stride[ndim-1] == element_size
    mov rax, [r12 + TENSOR_STRIDE]
    mov rcx, r13
    dec rcx
    cmp [rax + rcx*8], rbx
    jne .is_contig_no
    
    ; Check each stride[i] == stride[i+1] * shape[i+1]
    test rcx, rcx
    jz .is_contig_yes               ; only 1 dim

.is_contig_loop:
    dec rcx
    js .is_contig_yes
    
    mov rax, [r12 + TENSOR_STRIDE]
    mov rdx, [r12 + TENSOR_SHAPE]
    
    mov rsi, [rax + rcx*8 + 8]      ; stride[i+1]
    imul rsi, [rdx + rcx*8 + 8]     ; * shape[i+1]
    cmp [rax + rcx*8], rsi          ; stride[i]
    jne .is_contig_no
    jmp .is_contig_loop

.is_contig_yes:
    mov eax, 1
    jmp .is_contig_done

.is_contig_no:
    xor eax, eax
    jmp .is_contig_done

.is_contig_null:
    mov eax, -1

.is_contig_done:
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_tensor_make_contiguous - Return contiguous copy if needed
; Arguments: NeuralTensor* tensor
; Returns: NeuralTensor* (may be same tensor if already contiguous, or new copy)
; =============================================================================
neural_tensor_make_contiguous:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 8
    
    test rdi, rdi
    jz .make_contig_null
    
    mov r12, rdi
    
    ; Check if already contiguous
    call neural_tensor_is_contiguous
    cmp eax, 1
    je .make_contig_same
    
    ; Create contiguous copy
    mov rdi, [r12 + TENSOR_NDIM]
    mov rsi, [r12 + TENSOR_SHAPE]
    mov edx, [r12 + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .make_contig_alloc_fail
    mov rbx, rax
    
    ; Copy data (handles non-contiguous source)
    mov rdi, rbx
    mov rsi, r12
    call tensor_copy
    
    mov rax, rbx
    jmp .make_contig_done

.make_contig_same:
    mov rax, r12
    jmp .make_contig_done

.make_contig_null:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax
    jmp .make_contig_done

.make_contig_alloc_fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY
    xor eax, eax

.make_contig_done:
    add rsp, 8
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_buffer_info - Get buffer protocol info for NumPy integration
; Arguments:
;   RDI = NeuralTensor* tensor
;   RSI = NeuralBufferInfo* info (output struct to fill)
; Returns: int (error code)
; 
; NeuralBufferInfo struct layout:
;   void* data          (offset 0)
;   uint64_t itemsize   (offset 8)
;   uint64_t ndim       (offset 16)
;   uint64_t* shape     (offset 24)
;   int64_t* strides    (offset 32)
;   int readonly        (offset 40)
;   char format[8]      (offset 44)
; =============================================================================
neural_buffer_info:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    test rdi, rdi
    jz .bufinfo_null
    test rsi, rsi
    jz .bufinfo_null
    
    mov r12, rdi            ; tensor
    mov rbx, rsi            ; info struct
    
    ; data pointer
    mov rax, [r12 + TENSOR_DATA]
    mov [rbx], rax
    
    ; itemsize (element size)
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, 0
    je .bufinfo_f32
    mov qword [rbx + 8], 8          ; float64
    mov byte [rbx + 44], 'd'        ; format 'd' for double
    jmp .bufinfo_cont
.bufinfo_f32:
    mov qword [rbx + 8], 4          ; float32
    mov byte [rbx + 44], 'f'        ; format 'f' for float
.bufinfo_cont:
    mov byte [rbx + 45], 0          ; null terminate format
    
    ; ndim
    mov rax, [r12 + TENSOR_NDIM]
    mov [rbx + 16], rax
    
    ; shape pointer
    mov rax, [r12 + TENSOR_SHAPE]
    mov [rbx + 24], rax
    
    ; strides pointer
    mov rax, [r12 + TENSOR_STRIDE]
    mov [rbx + 32], rax
    
    ; readonly flag (0 = writable)
    mov dword [rbx + 40], 0
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .bufinfo_done

.bufinfo_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.bufinfo_done:
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_tensor_copy - Copy tensor data
; Arguments:
;   RDI = NeuralTensor* dst
;   RSI = const NeuralTensor* src
; Returns: int (error code)
; =============================================================================
neural_tensor_copy:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null

    mov r12, rdi                    ; dst
    mov r13, rsi                    ; src

    ; Validate dtype compatibility
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, [r13 + TENSOR_DTYPE]
    jne .dtype_mismatch

    ; Validate element count compatibility
    mov rdi, r12
    call tensor_numel
    mov rbx, rax

    mov rdi, r13
    call tensor_numel
    cmp rax, rbx
    jne .shape_mismatch

    ; Copy data bytes from src into dst
    mov rdi, r12
    call tensor_data_size
    mov rdx, rax

    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    call mem_copy
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.shape_mismatch:
    mov eax, NEURAL_ERR_SHAPE_MISMATCH
    mov [rel last_error], eax
    jmp .done

.dtype_mismatch:
    mov eax, NEURAL_ERR_DTYPE_MISMATCH
    mov [rel last_error], eax
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_add - Element-wise addition
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* a
;   RDX = const NeuralTensor* b
; Returns: int (error code)
; =============================================================================
neural_add:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    ; Validate pointers
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    test rdx, rdx
    jz .null
    
    ; Get output data and size
    mov rdi, [r13 + TENSOR_DATA]   ; a->data
    mov rsi, [r14 + TENSOR_DATA]   ; b->data
    mov rdx, [r12 + TENSOR_DATA]   ; out->data
    
    ; Get number of elements
    mov rcx, r13
    push rdi
    push rsi
    push rdx
    mov rdi, r13
    call tensor_numel
    pop rdx
    pop rsi
    pop rdi
    mov rcx, rax
    
    ; Call kernel
    call ew_add
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_sub - Element-wise subtraction
; =============================================================================
neural_sub:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    test rdx, rdx
    jz .null
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r14 + TENSOR_DATA]
    mov rdx, [r12 + TENSOR_DATA]
    
    push rdi
    push rsi
    push rdx
    mov rdi, r13
    call tensor_numel
    pop rdx
    pop rsi
    pop rdi
    mov rcx, rax
    
    call ew_sub
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_mul - Element-wise multiplication
; =============================================================================
neural_mul:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    test rdx, rdx
    jz .null
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r14 + TENSOR_DATA]
    mov rdx, [r12 + TENSOR_DATA]
    
    push rdi
    push rsi
    push rdx
    mov rdi, r13
    call tensor_numel
    pop rdx
    pop rsi
    pop rdi
    mov rcx, rax
    
    call ew_mul
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_div - Element-wise division
; =============================================================================
neural_div:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    test rdx, rdx
    jz .null
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r14 + TENSOR_DATA]
    mov rdx, [r12 + TENSOR_DATA]
    
    push rdi
    push rsi
    push rdx
    mov rdi, r13
    call tensor_numel
    pop rdx
    pop rsi
    pop rdi
    mov rcx, rax
    
    call ew_div
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_matmul - Matrix multiplication
; =============================================================================
neural_matmul:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    test rdx, rdx
    jz .null
    
    call matmul
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop rbp
    ret

; =============================================================================
; neural_relu - ReLU activation
; =============================================================================
neural_relu:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null

    ; Save output tensor pointer
    mov r12, rdi

    ; Create input node from input tensor
    mov rdi, rsi
    xor esi, esi
    call node_create
    test rax, rax
    jz .internal
    mov r13, rax

    ; Compute ReLU node
    mov rdi, r13
    call node_relu
    test rax, rax
    jz .internal

    ; Copy output node tensor into caller-provided output tensor
    mov rsi, [rax]
    mov rdi, r12
    call neural_tensor_copy
    test eax, eax
    jnz .tensor_error
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.tensor_error:
    mov [rel last_error], eax
    jmp .done

.internal:
    mov eax, NEURAL_ERR_INTERNAL
    mov [rel last_error], eax
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_sigmoid - Sigmoid activation
; =============================================================================
neural_sigmoid:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null

    ; Save output tensor pointer
    mov r12, rdi

    ; Create input node from input tensor
    mov rdi, rsi
    xor esi, esi
    call node_create
    test rax, rax
    jz .internal
    mov r13, rax

    ; Compute sigmoid node
    mov rdi, r13
    call node_sigmoid
    test rax, rax
    jz .internal

    ; Copy output node tensor into caller-provided output tensor
    mov rsi, [rax]
    mov rdi, r12
    call neural_tensor_copy
    test eax, eax
    jnz .tensor_error
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.tensor_error:
    mov [rel last_error], eax
    jmp .done

.internal:
    mov eax, NEURAL_ERR_INTERNAL
    mov [rel last_error], eax
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_softmax - Softmax activation
; =============================================================================
neural_softmax:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null

    ; Save output tensor pointer
    mov r12, rdi

    ; Create input node from input tensor
    mov rdi, rsi
    xor esi, esi
    call node_create
    test rax, rax
    jz .internal
    mov r13, rax

    ; Compute softmax node
    mov rdi, r13
    call node_softmax
    test rax, rax
    jz .internal

    ; Copy output node tensor into caller-provided output tensor
    mov rsi, [rax]
    mov rdi, r12
    call neural_tensor_copy
    test eax, eax
    jnz .tensor_error
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.tensor_error:
    mov [rel last_error], eax
    jmp .done

.internal:
    mov eax, NEURAL_ERR_INTERNAL
    mov [rel last_error], eax
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_gelu - GELU activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
; Returns: int (error code)
; =============================================================================
neural_gelu:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .gelu_api_null
    test rsi, rsi
    jz .gelu_api_null
    
    call gelu_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .gelu_api_done

.gelu_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.gelu_api_done:
    pop rbp
    ret

; =============================================================================
; neural_leaky_relu - Leaky ReLU activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
;   XMM0 = double alpha (negative slope, e.g. 0.01)
; Returns: int (error code)
; =============================================================================
neural_leaky_relu:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .leaky_api_null
    test rsi, rsi
    jz .leaky_api_null
    
    call leaky_relu_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .leaky_api_done

.leaky_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.leaky_api_done:
    pop rbp
    ret

; =============================================================================
; neural_elu - ELU activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
;   XMM0 = double alpha (default 1.0)
; Returns: int (error code)
; =============================================================================
neural_elu:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .elu_api_null
    test rsi, rsi
    jz .elu_api_null
    
    call elu_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .elu_api_done

.elu_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.elu_api_done:
    pop rbp
    ret

; =============================================================================
; neural_selu - SELU activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
; Returns: int (error code)
; =============================================================================
neural_selu:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .selu_api_null
    test rsi, rsi
    jz .selu_api_null
    
    call selu_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .selu_api_done

.selu_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.selu_api_done:
    pop rbp
    ret

; =============================================================================
; neural_swish - Swish/SiLU activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
; Returns: int (error code)
; =============================================================================
neural_swish:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .swish_api_null
    test rsi, rsi
    jz .swish_api_null
    
    call swish_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .swish_api_done

.swish_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.swish_api_done:
    pop rbp
    ret

; =============================================================================
; neural_mish - Mish activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
; Returns: int (error code)
; =============================================================================
neural_mish:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .mish_api_null
    test rsi, rsi
    jz .mish_api_null
    
    call mish_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .mish_api_done

.mish_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.mish_api_done:
    pop rbp
    ret

; =============================================================================
; neural_hardswish - Hard Swish activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
; Returns: int (error code)
; =============================================================================
neural_hardswish:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .hardswish_api_null
    test rsi, rsi
    jz .hardswish_api_null
    
    call hardswish_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .hardswish_api_done

.hardswish_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.hardswish_api_done:
    pop rbp
    ret

; =============================================================================
; neural_softplus - Softplus activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
; Returns: int (error code)
; =============================================================================
neural_softplus:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .softplus_api_null
    test rsi, rsi
    jz .softplus_api_null
    
    call softplus_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .softplus_api_done

.softplus_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.softplus_api_done:
    pop rbp
    ret

; =============================================================================
; neural_hardtanh - Hardtanh activation
; Arguments:
;   RDI = NeuralTensor* out
;   RSI = const NeuralTensor* x
; Returns: int (error code)
; =============================================================================
neural_hardtanh:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .hardtanh_api_null
    test rsi, rsi
    jz .hardtanh_api_null
    
    call hardtanh_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .hardtanh_api_done

.hardtanh_api_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.hardtanh_api_done:
    pop rbp
    ret

; =============================================================================
; neural_clip_grad_norm - Clip gradients by global L2 norm
; Arguments:
;   RDI = NeuralOptimizer* opt
;   XMM0 = max_norm (double)
; Returns: int (error code)
; =============================================================================
neural_clip_grad_norm:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .clip_norm_null
    
    call clip_grad_norm_
    
    mov dword [rel last_error], NEURAL_OK
    jmp .clip_norm_done

.clip_norm_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.clip_norm_done:
    pop rbp
    ret

; =============================================================================
; neural_clip_grad_value - Clip gradient values to range [min_val, max_val]
; Arguments:
;   RDI = NeuralOptimizer* opt
;   XMM0 = min_val (double)
;   XMM1 = max_val (double)
; Returns: int (error code)
; =============================================================================
neural_clip_grad_value:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .clip_value_null
    
    call clip_grad_value_
    
    mov dword [rel last_error], NEURAL_OK
    jmp .clip_value_done

.clip_value_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.clip_value_done:
    pop rbp
    ret

; =============================================================================
; neural_linear_create - Create linear layer
; Arguments:
;   RDI = uint64_t in_features
;   RSI = uint64_t out_features
;   EDX = int bias
; Returns: NeuralLinear*
; =============================================================================
neural_linear_create:
    push rbp
    mov rbp, rsp
    
    call linear_create
    
    test rax, rax
    jz .fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY

.done:
    pop rbp
    ret

; =============================================================================
; neural_linear_free - Free linear layer
; =============================================================================
neural_linear_free:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .done
    
    ; linear_free may not exist - just free the memory directly
    call mem_free
    
.done:
    pop rbp
    ret

; =============================================================================
; neural_linear_forward - Forward pass through linear layer
; =============================================================================
neural_linear_forward:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    test rdx, rdx
    jz .null

    ; Save pointers
    mov r13, rdi                   ; layer/module pointer
    mov r12, rdx

    ; Create input node from input tensor
    mov rdi, rsi
    xor esi, esi
    call node_create
    test rax, rax
    jz .internal

    ; linear_forward expects (Module*, Node*) and returns Node*
    mov rdi, r13
    mov rsi, rax
    call linear_forward
    test rax, rax
    jz .internal

    ; Copy output node tensor into caller-provided output tensor
    mov rsi, [rax]
    mov rdi, r12
    call neural_tensor_copy
    test eax, eax
    jnz .tensor_error
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.tensor_error:
    mov [rel last_error], eax
    jmp .done

.internal:
    mov eax, NEURAL_ERR_INTERNAL
    mov [rel last_error], eax
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; neural_sgd_create - Create SGD optimizer
; Arguments:
;   XMM0 = double learning_rate
;   XMM1 = double momentum
; Returns: NeuralOptimizer*
; =============================================================================
neural_sgd_create:
    push rbp
    mov rbp, rsp
    
    ; Set up arguments for sgd_create: params=NULL, param_nodes=NULL, n_params=0
    xor rdi, rdi                    ; params = NULL
    xor rsi, rsi                    ; param_nodes = NULL
    xor edx, edx                    ; n_params = 0
    ; XMM0 and XMM1 already contain lr and momentum
    
    call sgd_create
    
    test rax, rax
    jz .fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY

.done:
    pop rbp
    ret

; =============================================================================
; neural_adam_create - Create Adam optimizer
; =============================================================================
neural_adam_create:
    push rbp
    mov rbp, rsp
    
    call adam_create
    
    test rax, rax
    jz .fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY

.done:
    pop rbp
    ret

; =============================================================================
; neural_adamw_create - Create AdamW optimizer
; =============================================================================
neural_adamw_create:
    push rbp
    mov rbp, rsp
    
    call adamw_create
    
    test rax, rax
    jz .fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY

.done:
    pop rbp
    ret

; =============================================================================
; neural_optimizer_free - Free optimizer
; =============================================================================
neural_optimizer_free:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .done
    
    call optimizer_free
    
.done:
    pop rbp
    ret

; =============================================================================
; neural_dataset_load_csv - Load dataset from CSV
; =============================================================================
neural_dataset_load_csv:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call dataset_load_csv
    
    test rax, rax
    jz .fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax
    jmp .done

.fail:
    mov dword [rel last_error], NEURAL_ERR_FILE_NOT_FOUND

.done:
    pop rbp
    ret

; =============================================================================
; neural_dataset_free - Free dataset
; =============================================================================
neural_dataset_free:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .done
    
    call dataset_free
    
.done:
    pop rbp
    ret

; =============================================================================
; neural_dataset_size - Get dataset size
; =============================================================================
neural_dataset_size:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call dataset_size
    jmp .done

.null:
    xor eax, eax

.done:
    pop rbp
    ret

; =============================================================================
; neural_config_load - Load configuration
; =============================================================================
neural_config_load:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call config_parse
    
    test rax, rax
    jz .fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax
    jmp .done

.fail:
    mov dword [rel last_error], NEURAL_ERR_FILE_NOT_FOUND

.done:
    pop rbp
    ret

; =============================================================================
; neural_config_free - Free configuration
; =============================================================================
neural_config_free:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .done
    
    call config_free
    
.done:
    pop rbp
    ret

; =============================================================================
; neural_get_simd_level - Get SIMD level
; Returns: int (0=scalar, 1=SSE2, 2=AVX, 3=AVX2, 4=AVX-512)
; =============================================================================
neural_get_simd_level:
    push rbp
    mov rbp, rsp
    
    call detect_cpu_features
    
    pop rbp
    ret

; =============================================================================
; neural_get_simd_name - Get SIMD level name
; Returns: const char*
; =============================================================================
neural_get_simd_name:
    push rbp
    mov rbp, rsp
    
    call detect_cpu_features
    
    cmp eax, 0
    je .scalar
    cmp eax, 1
    je .sse2
    cmp eax, 2
    je .avx
    cmp eax, 3
    je .avx2
    cmp eax, 4
    je .avx512
    
.scalar:
    lea rax, [rel simd_name_scalar]
    jmp .done
.sse2:
    lea rax, [rel simd_name_sse2]
    jmp .done
.avx:
    lea rax, [rel simd_name_avx]
    jmp .done
.avx2:
    lea rax, [rel simd_name_avx2]
    jmp .done
.avx512:
    lea rax, [rel simd_name_avx512]

.done:
    pop rbp
    ret

; =============================================================================
; Stub implementations for functions that need more complex handling
; =============================================================================

neural_tensor_random:
    ; TODO: Implement random tensor creation
    mov dword [rel last_error], NEURAL_ERR_NOT_IMPLEMENTED
    xor eax, eax
    ret

; =============================================================================
; neural_tensor_from_data - Create tensor by copying data from buffer
; Arguments:
;   RDI = void* data (source buffer to copy from)
;   RSI = const uint64_t* shape
;   RDX = uint64_t ndim
;   ECX = int dtype
; Returns: NeuralTensor* (NULL on error)
; =============================================================================
neural_tensor_from_data:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    ; Validate arguments
    test rdi, rdi
    jz .from_data_null
    test rsi, rsi
    jz .from_data_null
    
    mov r12, rdi            ; source data
    mov r13, rsi            ; shape
    mov r14, rdx            ; ndim
    mov r15d, ecx           ; dtype
    
    ; Create tensor with given shape
    mov rdi, r14            ; ndim
    mov rsi, r13            ; shape
    mov edx, r15d           ; dtype
    call tensor_create
    test rax, rax
    jz .from_data_alloc_fail
    mov rbx, rax            ; tensor
    
    ; Calculate bytes to copy
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax            ; numel
    
    ; Determine element size
    cmp r15d, 0             ; DT_FLOAT32
    je .from_data_f32
    mov rax, 8              ; DT_FLOAT64
    jmp .from_data_copy
.from_data_f32:
    mov rax, 4
.from_data_copy:
    imul rcx, rax           ; total bytes
    
    ; Copy data
    mov rdi, [rbx + TENSOR_DATA]
    mov rsi, r12            ; source
    mov rdx, rcx            ; bytes
    call mem_copy
    
    mov dword [rel last_error], NEURAL_OK
    mov rax, rbx
    jmp .from_data_done

.from_data_null:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax
    jmp .from_data_done

.from_data_alloc_fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY
    xor eax, eax

.from_data_done:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_tensor_from_buffer - Create tensor wrapping external buffer (zero-copy)
; This creates a tensor that directly uses the provided buffer without copying.
; The caller is responsible for keeping the buffer alive while the tensor exists.
; Arguments:
;   RDI = void* buffer (external buffer to wrap - must remain valid)
;   RSI = const uint64_t* shape
;   RDX = uint64_t ndim
;   ECX = int dtype
;   R8  = const int64_t* strides (bytes, or NULL for C-contiguous)
; Returns: NeuralTensor* (NULL on error)
; =============================================================================
neural_tensor_from_buffer:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    ; Validate arguments
    test rdi, rdi
    jz .from_buf_null
    test rsi, rsi
    jz .from_buf_null
    
    mov r12, rdi            ; buffer
    mov r13, rsi            ; shape
    mov r14, rdx            ; ndim
    mov r15d, ecx           ; dtype
    mov [rsp], r8           ; strides (or NULL)
    
    ; Allocate tensor struct
    mov rdi, 64             ; TENSOR_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .from_buf_alloc_fail
    mov rbx, rax
    
    ; Set data pointer directly (zero-copy)
    mov [rbx + TENSOR_DATA], r12
    
    ; Set ndim and dtype
    mov [rbx + TENSOR_NDIM], r14
    mov [rbx + TENSOR_DTYPE], r15d
    
    ; Allocate and copy shape
    lea rdi, [r14*8]
    mov rsi, 8
    call mem_alloc_aligned
    test rax, rax
    jz .from_buf_cleanup1
    mov [rbx + TENSOR_SHAPE], rax
    
    ; Copy shape values
    mov rdi, rax
    mov rsi, r13
    lea rdx, [r14*8]
    call mem_copy
    
    ; Allocate stride array
    lea rdi, [r14*8]
    mov rsi, 8
    call mem_alloc_aligned
    test rax, rax
    jz .from_buf_cleanup2
    mov [rbx + TENSOR_STRIDE], rax
    mov r8, rax             ; stride ptr
    
    ; Check if custom strides provided
    mov rcx, [rsp]          ; original strides arg
    test rcx, rcx
    jnz .from_buf_custom_stride
    
    ; Compute C-contiguous strides (row-major)
    ; Element size first
    cmp r15d, 0
    je .from_buf_stride_f32
    mov rax, 8
    jmp .from_buf_compute_stride
.from_buf_stride_f32:
    mov rax, 4
.from_buf_compute_stride:
    ; stride[ndim-1] = element_size
    mov rcx, r14
    dec rcx
    mov [r8 + rcx*8], rax
    
    test rcx, rcx
    jz .from_buf_stride_done
    
.from_buf_stride_loop:
    dec rcx
    js .from_buf_stride_done
    mov rax, [r8 + rcx*8 + 8]       ; stride[i+1]
    mov rdx, [rbx + TENSOR_SHAPE]
    imul rax, [rdx + rcx*8 + 8]     ; * shape[i+1]
    mov [r8 + rcx*8], rax
    jmp .from_buf_stride_loop

.from_buf_custom_stride:
    ; Copy provided strides (already in bytes)
    mov rdi, r8
    mov rsi, [rsp]
    lea rdx, [r14*8]
    call mem_copy

.from_buf_stride_done:
    ; Set flags to mark this as external buffer (don't free data on tensor_free)
    mov dword [rbx + TENSOR_FLAGS], 1   ; TENSOR_FLAG_EXTERNAL_BUFFER
    
    mov dword [rel last_error], NEURAL_OK
    mov rax, rbx
    jmp .from_buf_done

.from_buf_null:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax
    jmp .from_buf_done

.from_buf_alloc_fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY
    xor eax, eax
    jmp .from_buf_done

.from_buf_cleanup2:
    mov rdi, [rbx + TENSOR_SHAPE]
    call mem_free
.from_buf_cleanup1:
    mov rdi, rbx
    call mem_free
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY
    xor eax, eax

.from_buf_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

neural_tensor_reshape:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call tensor_reshape
    jmp .done

.null:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax

.done:
    pop rbp
    ret

neural_sum:
    mov dword [rel last_error], NEURAL_ERR_NOT_IMPLEMENTED
    mov eax, NEURAL_ERR_NOT_IMPLEMENTED
    ret

neural_mean:
    mov dword [rel last_error], NEURAL_ERR_NOT_IMPLEMENTED
    mov eax, NEURAL_ERR_NOT_IMPLEMENTED
    ret

neural_tanh:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    
    call tanh_forward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop rbp
    ret

neural_linear_weight:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    ; Get weight tensor from params[0]
    mov rax, [rdi + MODULE_PARAMS]
    test rax, rax
    jz .null
    mov rax, [rax]                 ; params[0] = weight
    jmp .done

.null:
    xor eax, eax

.done:
    pop rbp
    ret

neural_linear_bias:
    push rbp
    mov rbp, rsp

    test rdi, rdi
    jz .null

    ; Get bias tensor from params[1]
    mov rax, [rdi + MODULE_PARAMS]
    test rax, rax
    jz .null
    mov rax, [rax + 8]             ; params[1] = bias
    jmp .done

.null:
    xor eax, eax

.done:
    pop rbp
    ret

neural_mse_loss:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .mse_null
    test rsi, rsi
    jz .mse_null
    test rdx, rdx
    jz .mse_null
    
    call mse_loss
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .mse_done

.mse_null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.mse_done:
    pop rbp
    ret

neural_cross_entropy_loss:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    test rdx, rdx
    jz .null
    
    call cross_entropy_loss
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop rbp
    ret

neural_optimizer_step:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call optimizer_step
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop rbp
    ret

neural_optimizer_zero_grad:
    ; Zero out gradient tensors
    mov dword [rel last_error], NEURAL_ERR_NOT_IMPLEMENTED
    mov eax, NEURAL_ERR_NOT_IMPLEMENTED
    ret

; =============================================================================
; Node/Autograd API Functions
; =============================================================================

neural_node_create:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call node_create
    
    test rax, rax
    jz .fail
    
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov dword [rel last_error], NEURAL_ERR_NULL_POINTER
    xor eax, eax
    jmp .done

.fail:
    mov dword [rel last_error], NEURAL_ERR_OUT_OF_MEMORY

.done:
    pop rbp
    ret

neural_node_free:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .done
    
    call node_free
    
.done:
    pop rbp
    ret

neural_backward:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call backward
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop rbp
    ret

neural_node_grad:
    ; TODO: Get gradient from node
    xor eax, eax
    ret

neural_dataset_get_batch:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call dataset_get_batch
    
    xor eax, eax
    mov dword [rel last_error], NEURAL_OK
    jmp .done

.null:
    mov eax, NEURAL_ERR_NULL_POINTER
    mov [rel last_error], eax

.done:
    pop rbp
    ret

neural_config_get_int:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call config_get_int
    jmp .done

.null:
    mov eax, ecx    ; Return default value

.done:
    pop rbp
    ret

neural_config_get_float:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call config_get_float
    jmp .done

.null:
    movsd xmm0, xmm2    ; Return default value

.done:
    pop rbp
    ret

neural_config_get_string:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    call config_get_string
    jmp .done

.null:
    mov rax, rcx    ; Return default value

.done:
    pop rbp
    ret
