; =============================================================================
; tensor.asm - Tensor Subsystem for Deep Learning Framework
; =============================================================================
; Tensor descriptor, creation, manipulation functions
; =============================================================================

; Tensor struct layout (64 bytes total):
; Offset  Size    Field
; 0       8       data        (void*)
; 8       8       ndim        (uint64_t)
; 16      8       shape       (uint64_t*)
; 24      8       stride      (uint64_t*)
; 32      4       dtype       (uint32_t)
; 36      4       flags       (uint32_t)
; 40      24      padding     (for alignment)

; Constants
%define TENSOR_SIZE         64
%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_STRIDE       24
%define TENSOR_DTYPE        32
%define TENSOR_FLAGS        36

; Dtype constants
%define DT_FLOAT32          0
%define DT_FLOAT64          1

; Flag constants
%define FLAG_OWN_DATA       1
%define FLAG_OWN_META       2

; Byte sizes per dtype
%define SIZEOF_FLOAT32      4
%define SIZEOF_FLOAT64      8

section .data
    align 8
    err_null_tensor:    db "Error: Null tensor pointer", 0
    err_null_shape:     db "Error: Null shape pointer", 0
    err_invalid_dtype:  db "Error: Invalid dtype", 0
    err_shape_mismatch: db "Error: Shape mismatch in reshape", 0
    err_alloc_failed:   db "Error: Tensor allocation failed", 0

section .bss
    align 8

section .text

; External functions
extern mem_alloc
extern mem_alloc_aligned
extern mem_free
extern mem_zero
extern mem_copy
extern panic
extern assert_true

; Export tensor functions
global tensor_create
global tensor_free
global tensor_zeros
global tensor_copy
global tensor_fill
global tensor_numel
global tensor_reshape
global tensor_view
global tensor_get_dtype_size
global tensor_data_size

; Export constants (as functions that return values)
global get_dt_float32
global get_dt_float64

; =============================================================================
; get_dt_float32 - Return DT_FLOAT32 constant
; Returns: RAX = 0
; =============================================================================
get_dt_float32:
    xor eax, eax
    ret

; =============================================================================
; get_dt_float64 - Return DT_FLOAT64 constant
; Returns: RAX = 1
; =============================================================================
get_dt_float64:
    mov eax, 1
    ret

; =============================================================================
; tensor_get_dtype_size - Get byte size of dtype
; Arguments:
;   RDI = dtype (uint32_t)
; Returns:
;   RAX = byte size (4 for float32, 8 for float64)
; =============================================================================
tensor_get_dtype_size:
    cmp edi, DT_FLOAT32
    je .float32
    cmp edi, DT_FLOAT64
    je .float64
    ; Invalid dtype - return 0
    xor eax, eax
    ret
.float32:
    mov eax, SIZEOF_FLOAT32
    ret
.float64:
    mov eax, SIZEOF_FLOAT64
    ret

; =============================================================================
; tensor_numel - Get total number of elements in tensor
; Arguments:
;   RDI = Tensor* tensor
; Returns:
;   RAX = number of elements
; =============================================================================
tensor_numel:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    ; Check for null
    test rdi, rdi
    jz .null_tensor
    
    mov r12, rdi                    ; Save tensor pointer
    mov rcx, [r12 + TENSOR_NDIM]    ; ndim
    mov rsi, [r12 + TENSOR_SHAPE]   ; shape pointer
    
    ; If ndim == 0, return 1 (scalar)
    test rcx, rcx
    jz .scalar
    
    ; Multiply all dimensions
    mov rax, 1
    xor rbx, rbx                    ; index = 0
.numel_loop:
    cmp rbx, rcx
    jge .numel_done
    imul rax, [rsi + rbx*8]
    inc rbx
    jmp .numel_loop

.numel_done:
    pop r12
    pop rbx
    pop rbp
    ret

.scalar:
    mov rax, 1
    pop r12
    pop rbx
    pop rbp
    ret

.null_tensor:
    xor eax, eax
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_data_size - Get total data size in bytes
; Arguments:
;   RDI = Tensor* tensor
; Returns:
;   RAX = size in bytes
; =============================================================================
tensor_data_size:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8
    
    mov rbx, rdi                    ; Save tensor pointer
    
    ; Get numel
    call tensor_numel
    mov rcx, rax                    ; numel
    
    ; Get dtype size
    mov edi, [rbx + TENSOR_DTYPE]
    call tensor_get_dtype_size
    
    ; Multiply
    imul rax, rcx
    
    add rsp, 8
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_create - Create a new tensor
; Arguments:
;   RDI = ndim (uint64_t)
;   RSI = shape* (pointer to array of dimensions)
;   RDX = dtype (uint32_t)
; Returns:
;   RAX = Tensor* (or NULL on failure)
; =============================================================================
tensor_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24                     ; Local storage + alignment
    
    ; Save arguments
    mov r12, rdi                    ; ndim
    mov r13, rsi                    ; shape pointer
    mov r14d, edx                   ; dtype
    
    ; Validate dtype
    cmp r14d, DT_FLOAT64
    ja .invalid_dtype
    
    ; Allocate tensor struct (64 bytes, 16-byte aligned)
    mov rdi, TENSOR_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed
    mov r15, rax                    ; tensor pointer
    
    ; Initialize tensor struct
    mov qword [r15 + TENSOR_DATA], 0
    mov [r15 + TENSOR_NDIM], r12
    mov qword [r15 + TENSOR_SHAPE], 0
    mov qword [r15 + TENSOR_STRIDE], 0
    mov [r15 + TENSOR_DTYPE], r14d
    mov dword [r15 + TENSOR_FLAGS], FLAG_OWN_DATA | FLAG_OWN_META
    
    ; Allocate shape array
    test r12, r12
    jz .no_dims                     ; Handle scalar (ndim=0)
    
    mov rdi, r12
    shl rdi, 3                      ; ndim * 8 bytes
    call mem_alloc
    test rax, rax
    jz .alloc_failed_cleanup
    mov [r15 + TENSOR_SHAPE], rax
    mov rbx, rax                    ; shape array pointer
    
    ; Copy shape values
    xor rcx, rcx
.copy_shape:
    cmp rcx, r12
    jge .shape_done
    mov rax, [r13 + rcx*8]
    mov [rbx + rcx*8], rax
    inc rcx
    jmp .copy_shape

.shape_done:
    ; Allocate stride array
    mov rdi, r12
    shl rdi, 3                      ; ndim * 8 bytes
    call mem_alloc
    test rax, rax
    jz .alloc_failed_cleanup
    mov [r15 + TENSOR_STRIDE], rax
    
    ; Compute row-major strides
    ; stride[ndim-1] = 1
    ; stride[i] = stride[i+1] * shape[i+1]
    mov rsi, [r15 + TENSOR_STRIDE]
    mov rdi, [r15 + TENSOR_SHAPE]
    
    mov rcx, r12
    dec rcx                         ; rcx = ndim - 1
    mov qword [rsi + rcx*8], 1      ; stride[ndim-1] = 1
    
    test rcx, rcx
    jz .strides_done                ; Only one dimension
    
.compute_strides:
    dec rcx
    js .strides_done
    ; stride[i] = stride[i+1] * shape[i+1]
    mov rax, [rsi + rcx*8 + 8]      ; stride[i+1]
    imul rax, [rdi + rcx*8 + 8]     ; * shape[i+1]
    mov [rsi + rcx*8], rax          ; stride[i]
    jmp .compute_strides

.strides_done:
.no_dims:
    ; Calculate total number of elements
    mov rdi, r15
    call tensor_numel
    mov rbx, rax                    ; numel
    
    ; Calculate data size
    mov edi, r14d
    call tensor_get_dtype_size
    imul rbx, rax                   ; total bytes
    
    ; Handle zero-size tensor
    test rbx, rbx
    jz .zero_size
    
    ; Allocate data with 32-byte alignment for AVX
    mov rdi, rbx
    mov rsi, 32
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed_cleanup
    mov [r15 + TENSOR_DATA], rax
    
    ; Zero the data
    mov rdi, rax
    mov rsi, rbx
    call mem_zero
    
.zero_size:
    ; Return tensor pointer
    mov rax, r15
    
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.invalid_dtype:
    lea rdi, [rel err_invalid_dtype]
    call panic

.alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed_cleanup:
    ; Free partially allocated tensor
    mov rdi, r15
    call tensor_free
    xor eax, eax
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_free - Free tensor and its data
; Arguments:
;   RDI = Tensor* tensor
; Returns:
;   nothing
; =============================================================================
tensor_free:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 8
    
    ; Check for null
    test rdi, rdi
    jz .done
    
    mov r12, rdi                    ; Save tensor pointer
    mov ebx, [r12 + TENSOR_FLAGS]   ; Get flags
    
    ; Free data if owned
    test ebx, FLAG_OWN_DATA
    jz .skip_data
    mov rdi, [r12 + TENSOR_DATA]
    test rdi, rdi
    jz .skip_data
    call mem_free

.skip_data:
    ; Free metadata if owned
    test ebx, FLAG_OWN_META
    jz .skip_meta
    
    ; Free shape
    mov rdi, [r12 + TENSOR_SHAPE]
    test rdi, rdi
    jz .skip_shape
    call mem_free
.skip_shape:
    
    ; Free stride
    mov rdi, [r12 + TENSOR_STRIDE]
    test rdi, rdi
    jz .skip_stride
    call mem_free
.skip_stride:

.skip_meta:
    ; Free tensor struct
    mov rdi, r12
    call mem_free

.done:
    add rsp, 8
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_zeros - Create a tensor filled with zeros
; Arguments:
;   RDI = ndim (uint64_t)
;   RSI = shape* (pointer to array of dimensions)
;   RDX = dtype (uint32_t)
; Returns:
;   RAX = Tensor*
; =============================================================================
tensor_zeros:
    ; tensor_create already zeros the data
    jmp tensor_create

; =============================================================================
; tensor_copy - Create a deep copy of tensor
; Arguments:
;   RDI = Tensor* src
; Returns:
;   RAX = Tensor* (new copy)
; =============================================================================
tensor_copy:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    ; Check for null
    test rdi, rdi
    jz .null_src
    
    mov r12, rdi                    ; Save src tensor
    
    ; Create new tensor with same shape
    mov rdi, [r12 + TENSOR_NDIM]
    mov rsi, [r12 + TENSOR_SHAPE]
    mov edx, [r12 + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r13, rax                    ; new tensor
    
    ; Copy data
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r12 + TENSOR_DATA]
    
    ; Calculate data size
    push rdi
    push rsi
    mov rdi, r12
    call tensor_data_size
    mov rdx, rax
    pop rsi
    pop rdi
    
    call mem_copy
    
    mov rax, r13
    
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.null_src:
.alloc_failed:
    xor eax, eax
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_fill - Fill tensor with a value
; Arguments:
;   RDI = Tensor* tensor
;   XMM0 = value (as double)
; Returns:
;   nothing
; =============================================================================
tensor_fill:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi                    ; tensor pointer
    movsd [rsp], xmm0               ; Save value on stack
    
    ; Get number of elements
    call tensor_numel
    mov r13, rax                    ; numel
    
    test r13, r13
    jz .done
    
    ; Get data pointer
    mov rbx, [r12 + TENSOR_DATA]
    mov ecx, [r12 + TENSOR_DTYPE]
    movsd xmm0, [rsp]               ; Restore value
    
    cmp ecx, DT_FLOAT32
    je .fill_float32
    cmp ecx, DT_FLOAT64
    je .fill_float64
    jmp .done

.fill_float32:
    ; Convert double to float
    cvtsd2ss xmm0, xmm0
    xor rcx, rcx
.fill_f32_loop:
    cmp rcx, r13
    jge .done
    movss [rbx + rcx*4], xmm0
    inc rcx
    jmp .fill_f32_loop

.fill_float64:
    xor rcx, rcx
.fill_f64_loop:
    cmp rcx, r13
    jge .done
    movsd [rbx + rcx*8], xmm0
    inc rcx
    jmp .fill_f64_loop

.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_reshape - Reshape tensor (must have compatible number of elements)
; Arguments:
;   RDI = Tensor* tensor
;   RSI = ndim_new (uint64_t)
;   RDX = shape_new* (pointer to new shape array)
; Returns:
;   RAX = Tensor* (new view, shares data)
; =============================================================================
tensor_reshape:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    test rdi, rdi
    jz .null_tensor
    
    mov r12, rdi                    ; original tensor
    mov r13, rsi                    ; new ndim
    mov r14, rdx                    ; new shape
    
    ; Calculate original numel
    call tensor_numel
    mov r15, rax                    ; original numel
    
    ; Calculate new numel
    mov rax, 1
    test r13, r13
    jz .check_numel                 ; scalar case
    xor rcx, rcx
.new_numel_loop:
    cmp rcx, r13
    jge .check_numel
    imul rax, [r14 + rcx*8]
    inc rcx
    jmp .new_numel_loop

.check_numel:
    cmp rax, r15
    jne .shape_mismatch
    
    ; Create a view with new shape
    mov rdi, r12
    mov rsi, r13
    mov rdx, r14
    ; Pass NULL for stride to auto-compute
    xor rcx, rcx
    call tensor_view_internal
    
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.shape_mismatch:
    lea rdi, [rel err_shape_mismatch]
    call panic

.null_tensor:
    xor eax, eax
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_view_internal - Create view with optional custom strides
; Arguments:
;   RDI = Tensor* tensor (source)
;   RSI = ndim_new
;   RDX = shape_new*
;   RCX = stride_new* (or NULL to compute row-major)
; Returns:
;   RAX = Tensor* (view)
; =============================================================================
tensor_view_internal:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; source tensor
    mov r13, rsi                    ; new ndim
    mov r14, rdx                    ; new shape
    mov r15, rcx                    ; new stride (or NULL)
    
    ; Allocate tensor struct
    mov rdi, TENSOR_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .view_failed
    mov rbx, rax                    ; new tensor
    
    ; Copy data pointer (share data)
    mov rax, [r12 + TENSOR_DATA]
    mov [rbx + TENSOR_DATA], rax
    
    ; Set ndim
    mov [rbx + TENSOR_NDIM], r13
    
    ; Copy dtype
    mov eax, [r12 + TENSOR_DTYPE]
    mov [rbx + TENSOR_DTYPE], eax
    
    ; Set flags (don't own data, own metadata)
    mov dword [rbx + TENSOR_FLAGS], FLAG_OWN_META
    
    ; Allocate and copy shape
    test r13, r13
    jz .no_dims
    
    mov rdi, r13
    shl rdi, 3
    call mem_alloc
    test rax, rax
    jz .view_failed
    mov [rbx + TENSOR_SHAPE], rax
    
    ; Copy shape
    mov rdi, rax
    mov rsi, r14
    mov rdx, r13
    shl rdx, 3
    call mem_copy
    
    ; Allocate stride array
    mov rdi, r13
    shl rdi, 3
    call mem_alloc
    test rax, rax
    jz .view_failed
    mov [rbx + TENSOR_STRIDE], rax
    
    ; Copy or compute strides
    test r15, r15
    jz .compute_strides
    
    ; Copy provided strides
    mov rdi, rax
    mov rsi, r15
    mov rdx, r13
    shl rdx, 3
    call mem_copy
    jmp .view_done

.compute_strides:
    ; Compute row-major strides
    mov rsi, [rbx + TENSOR_STRIDE]
    mov rdi, [rbx + TENSOR_SHAPE]
    
    mov rcx, r13
    dec rcx
    mov qword [rsi + rcx*8], 1
    
    test rcx, rcx
    jz .view_done
    
.compute_stride_loop:
    dec rcx
    js .view_done
    mov rax, [rsi + rcx*8 + 8]
    imul rax, [rdi + rcx*8 + 8]
    mov [rsi + rcx*8], rax
    jmp .compute_stride_loop

.no_dims:
    mov qword [rbx + TENSOR_SHAPE], 0
    mov qword [rbx + TENSOR_STRIDE], 0

.view_done:
    mov rax, rbx
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.view_failed:
    xor eax, eax
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_view - Create a view of tensor with custom shape and strides
; Arguments:
;   RDI = Tensor* tensor (source)
;   RSI = ndim_new
;   RDX = shape_new*
;   RCX = stride_new*
; Returns:
;   RAX = Tensor* (view, shares data)
; =============================================================================
tensor_view:
    jmp tensor_view_internal
