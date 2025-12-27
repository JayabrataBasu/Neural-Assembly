; =============================================================================
; math_kernels.asm - Core BLAS-like Math Kernels with SIMD
; =============================================================================
; Elementwise operations, reductions, matrix multiply
; Uses AVX/SSE for vectorization
; =============================================================================

; Tensor struct offsets
%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_STRIDE       24
%define TENSOR_DTYPE        32
%define TENSOR_FLAGS        36

; Dtype constants
%define DT_FLOAT32          0
%define DT_FLOAT64          1

section .data
    align 8
    err_shape_mismatch: db "Error: Shape mismatch in operation", 0
    err_dtype_mismatch: db "Error: Dtype mismatch in operation", 0
    err_null_tensor:    db "Error: Null tensor in math operation", 0

section .bss
    align 32

section .text

; External functions
extern tensor_numel
extern tensor_get_dtype_size
extern panic
extern assert_true
extern mem_alloc
extern mem_free

; Export math kernel functions
global ew_add
global ew_sub
global ew_mul
global ew_div
global ew_scalar_add
global ew_scalar_mul
global ew_max
global ew_neg
global reduce_sum
global reduce_mean
global matmul
global tensor_transpose_2d

; =============================================================================
; Helper: Get tensor data pointer and element count
; =============================================================================

; =============================================================================
; ew_add - Elementwise addition: out = a + b
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* a
;   RDX = Tensor* b
; Returns:
;   nothing
; =============================================================================
ew_add:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; a
    mov r14, rdx                    ; b
    
    ; Get element count
    mov rdi, r12
    call tensor_numel
    mov r15, rax                    ; numel
    
    ; Get data pointers
    mov rdi, [r12 + TENSOR_DATA]    ; out data
    mov rsi, [r13 + TENSOR_DATA]    ; a data
    mov rdx, [r14 + TENSOR_DATA]    ; b data
    
    ; Check dtype
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .add_f32
    cmp eax, DT_FLOAT64
    je .add_f64
    jmp .done

.add_f32:
    ; Process 8 floats at a time with AVX
    mov rcx, r15
    shr rcx, 3                      ; rcx = numel / 8
    test rcx, rcx
    jz .add_f32_remainder
    
.add_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vmovups ymm1, [rel rdx]
    vaddps ymm0, ymm0, ymm1
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .add_f32_avx_loop

.add_f32_remainder:
    ; Handle remaining elements
    mov rcx, r15
    and rcx, 7                      ; remaining = numel % 8
    test rcx, rcx
    jz .done

.add_f32_scalar_loop:
    movss xmm0, [rel rsi]
    addss xmm0, [rel rdx]
    movss [rel rdi], xmm0
    add rsi, 4
    add rdx, 4
    add rdi, 4
    dec rcx
    jnz .add_f32_scalar_loop
    jmp .done

.add_f64:
    ; Process 4 doubles at a time with AVX
    mov rcx, r15
    shr rcx, 2                      ; rcx = numel / 4
    test rcx, rcx
    jz .add_f64_remainder

.add_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vmovupd ymm1, [rel rdx]
    vaddpd ymm0, ymm0, ymm1
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .add_f64_avx_loop

.add_f64_remainder:
    mov rcx, r15
    and rcx, 3
    test rcx, rcx
    jz .done

.add_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    addsd xmm0, [rel rdx]
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdx, 8
    add rdi, 8
    dec rcx
    jnz .add_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ew_sub - Elementwise subtraction: out = a - b
; =============================================================================
ew_sub:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    mov rdi, r12
    call tensor_numel
    mov r15, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    mov rdx, [r14 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .sub_f32
    cmp eax, DT_FLOAT64
    je .sub_f64
    jmp .done

.sub_f32:
    mov rcx, r15
    shr rcx, 3
    test rcx, rcx
    jz .sub_f32_remainder

.sub_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vmovups ymm1, [rel rdx]
    vsubps ymm0, ymm0, ymm1
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .sub_f32_avx_loop

.sub_f32_remainder:
    mov rcx, r15
    and rcx, 7
    test rcx, rcx
    jz .done

.sub_f32_scalar_loop:
    movss xmm0, [rel rsi]
    subss xmm0, [rel rdx]
    movss [rel rdi], xmm0
    add rsi, 4
    add rdx, 4
    add rdi, 4
    dec rcx
    jnz .sub_f32_scalar_loop
    jmp .done

.sub_f64:
    mov rcx, r15
    shr rcx, 2
    test rcx, rcx
    jz .sub_f64_remainder

.sub_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vmovupd ymm1, [rel rdx]
    vsubpd ymm0, ymm0, ymm1
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .sub_f64_avx_loop

.sub_f64_remainder:
    mov rcx, r15
    and rcx, 3
    test rcx, rcx
    jz .done

.sub_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    subsd xmm0, [rel rdx]
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdx, 8
    add rdi, 8
    dec rcx
    jnz .sub_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ew_mul - Elementwise multiplication: out = a * b
; =============================================================================
ew_mul:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    mov rdi, r12
    call tensor_numel
    mov r15, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    mov rdx, [r14 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .mul_f32
    cmp eax, DT_FLOAT64
    je .mul_f64
    jmp .done

.mul_f32:
    mov rcx, r15
    shr rcx, 3
    test rcx, rcx
    jz .mul_f32_remainder

.mul_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vmovups ymm1, [rel rdx]
    vmulps ymm0, ymm0, ymm1
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .mul_f32_avx_loop

.mul_f32_remainder:
    mov rcx, r15
    and rcx, 7
    test rcx, rcx
    jz .done

.mul_f32_scalar_loop:
    movss xmm0, [rel rsi]
    mulss xmm0, [rel rdx]
    movss [rel rdi], xmm0
    add rsi, 4
    add rdx, 4
    add rdi, 4
    dec rcx
    jnz .mul_f32_scalar_loop
    jmp .done

.mul_f64:
    mov rcx, r15
    shr rcx, 2
    test rcx, rcx
    jz .mul_f64_remainder

.mul_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vmovupd ymm1, [rel rdx]
    vmulpd ymm0, ymm0, ymm1
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .mul_f64_avx_loop

.mul_f64_remainder:
    mov rcx, r15
    and rcx, 3
    test rcx, rcx
    jz .done

.mul_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    mulsd xmm0, [rel rdx]
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdx, 8
    add rdi, 8
    dec rcx
    jnz .mul_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ew_div - Elementwise division: out = a / b
; =============================================================================
ew_div:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    mov rdi, r12
    call tensor_numel
    mov r15, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    mov rdx, [r14 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .div_f32
    cmp eax, DT_FLOAT64
    je .div_f64
    jmp .done

.div_f32:
    mov rcx, r15
    shr rcx, 3
    test rcx, rcx
    jz .div_f32_remainder

.div_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vmovups ymm1, [rel rdx]
    vdivps ymm0, ymm0, ymm1
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .div_f32_avx_loop

.div_f32_remainder:
    mov rcx, r15
    and rcx, 7
    test rcx, rcx
    jz .done

.div_f32_scalar_loop:
    movss xmm0, [rel rsi]
    divss xmm0, [rel rdx]
    movss [rel rdi], xmm0
    add rsi, 4
    add rdx, 4
    add rdi, 4
    dec rcx
    jnz .div_f32_scalar_loop
    jmp .done

.div_f64:
    mov rcx, r15
    shr rcx, 2
    test rcx, rcx
    jz .div_f64_remainder

.div_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vmovupd ymm1, [rel rdx]
    vdivpd ymm0, ymm0, ymm1
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .div_f64_avx_loop

.div_f64_remainder:
    mov rcx, r15
    and rcx, 3
    test rcx, rcx
    jz .done

.div_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    divsd xmm0, [rel rdx]
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdx, 8
    add rdi, 8
    dec rcx
    jnz .div_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ew_scalar_add - Add scalar to tensor: out = a + scalar
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* a
;   XMM0 = scalar (as double)
; =============================================================================
ew_scalar_add:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 24
    
    mov r12, rdi
    mov r13, rsi
    movsd [rel rsp], xmm0               ; Save scalar
    
    mov rdi, r12
    call tensor_numel
    mov rbx, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    movsd xmm1, [rel rsp]               ; scalar
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .scalar_add_f32
    cmp eax, DT_FLOAT64
    je .scalar_add_f64
    jmp .done

.scalar_add_f32:
    cvtsd2ss xmm1, xmm1             ; Convert to float
    vbroadcastss ymm1, xmm1         ; Broadcast scalar
    
    mov rcx, rbx
    shr rcx, 3
    test rcx, rcx
    jz .scalar_add_f32_remainder

.scalar_add_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vaddps ymm0, ymm0, ymm1
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdi, 32
    dec rcx
    jnz .scalar_add_f32_avx_loop

.scalar_add_f32_remainder:
    mov rcx, rbx
    and rcx, 7
    test rcx, rcx
    jz .done

.scalar_add_f32_scalar_loop:
    movss xmm0, [rel rsi]
    addss xmm0, xmm1
    movss [rel rdi], xmm0
    add rsi, 4
    add rdi, 4
    dec rcx
    jnz .scalar_add_f32_scalar_loop
    jmp .done

.scalar_add_f64:
    vbroadcastsd ymm1, xmm1
    
    mov rcx, rbx
    shr rcx, 2
    test rcx, rcx
    jz .scalar_add_f64_remainder

.scalar_add_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vaddpd ymm0, ymm0, ymm1
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdi, 32
    dec rcx
    jnz .scalar_add_f64_avx_loop

.scalar_add_f64_remainder:
    mov rcx, rbx
    and rcx, 3
    test rcx, rcx
    jz .done

.scalar_add_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    addsd xmm0, xmm1
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdi, 8
    dec rcx
    jnz .scalar_add_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 24
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ew_scalar_mul - Multiply tensor by scalar: out = a * scalar
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* a
;   XMM0 = scalar (as double)
; =============================================================================
ew_scalar_mul:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 24
    
    mov r12, rdi
    mov r13, rsi
    movsd [rel rsp], xmm0
    
    mov rdi, r12
    call tensor_numel
    mov rbx, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    movsd xmm1, [rel rsp]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .scalar_mul_f32
    cmp eax, DT_FLOAT64
    je .scalar_mul_f64
    jmp .done

.scalar_mul_f32:
    cvtsd2ss xmm1, xmm1
    vbroadcastss ymm1, xmm1
    
    mov rcx, rbx
    shr rcx, 3
    test rcx, rcx
    jz .scalar_mul_f32_remainder

.scalar_mul_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vmulps ymm0, ymm0, ymm1
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdi, 32
    dec rcx
    jnz .scalar_mul_f32_avx_loop

.scalar_mul_f32_remainder:
    mov rcx, rbx
    and rcx, 7
    test rcx, rcx
    jz .done

.scalar_mul_f32_scalar_loop:
    movss xmm0, [rel rsi]
    mulss xmm0, xmm1
    movss [rel rdi], xmm0
    add rsi, 4
    add rdi, 4
    dec rcx
    jnz .scalar_mul_f32_scalar_loop
    jmp .done

.scalar_mul_f64:
    vbroadcastsd ymm1, xmm1
    
    mov rcx, rbx
    shr rcx, 2
    test rcx, rcx
    jz .scalar_mul_f64_remainder

.scalar_mul_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vmulpd ymm0, ymm0, ymm1
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdi, 32
    dec rcx
    jnz .scalar_mul_f64_avx_loop

.scalar_mul_f64_remainder:
    mov rcx, rbx
    and rcx, 3
    test rcx, rcx
    jz .done

.scalar_mul_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    mulsd xmm0, xmm1
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdi, 8
    dec rcx
    jnz .scalar_mul_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 24
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ew_max - Elementwise max: out = max(a, b)
; =============================================================================
ew_max:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    mov rdi, r12
    call tensor_numel
    mov r15, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    mov rdx, [r14 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .max_f32
    cmp eax, DT_FLOAT64
    je .max_f64
    jmp .done

.max_f32:
    mov rcx, r15
    shr rcx, 3
    test rcx, rcx
    jz .max_f32_remainder

.max_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vmovups ymm1, [rel rdx]
    vmaxps ymm0, ymm0, ymm1
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .max_f32_avx_loop

.max_f32_remainder:
    mov rcx, r15
    and rcx, 7
    test rcx, rcx
    jz .done

.max_f32_scalar_loop:
    movss xmm0, [rel rsi]
    maxss xmm0, [rel rdx]
    movss [rel rdi], xmm0
    add rsi, 4
    add rdx, 4
    add rdi, 4
    dec rcx
    jnz .max_f32_scalar_loop
    jmp .done

.max_f64:
    mov rcx, r15
    shr rcx, 2
    test rcx, rcx
    jz .max_f64_remainder

.max_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vmovupd ymm1, [rel rdx]
    vmaxpd ymm0, ymm0, ymm1
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdx, 32
    add rdi, 32
    dec rcx
    jnz .max_f64_avx_loop

.max_f64_remainder:
    mov rcx, r15
    and rcx, 3
    test rcx, rcx
    jz .done

.max_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    maxsd xmm0, [rel rdx]
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdx, 8
    add rdi, 8
    dec rcx
    jnz .max_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ew_neg - Elementwise negation: out = -a
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* a
; =============================================================================
ew_neg:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    mov r12, rdi
    mov r13, rsi
    
    mov rdi, r12
    call tensor_numel
    mov rbx, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .neg_f32
    cmp eax, DT_FLOAT64
    je .neg_f64
    jmp .done

.neg_f32:
    ; Create sign mask for float32
    mov eax, 0x80000000
    movd xmm2, eax
    vbroadcastss ymm2, xmm2
    
    mov rcx, rbx
    shr rcx, 3
    test rcx, rcx
    jz .neg_f32_remainder

.neg_f32_avx_loop:
    vmovups ymm0, [rel rsi]
    vxorps ymm0, ymm0, ymm2
    vmovups [rel rdi], ymm0
    add rsi, 32
    add rdi, 32
    dec rcx
    jnz .neg_f32_avx_loop

.neg_f32_remainder:
    mov rcx, rbx
    and rcx, 7
    test rcx, rcx
    jz .done

.neg_f32_scalar_loop:
    movss xmm0, [rel rsi]
    xorps xmm0, xmm2
    movss [rel rdi], xmm0
    add rsi, 4
    add rdi, 4
    dec rcx
    jnz .neg_f32_scalar_loop
    jmp .done

.neg_f64:
    mov rax, 0x8000000000000000
    movq xmm2, rax
    vbroadcastsd ymm2, xmm2
    
    mov rcx, rbx
    shr rcx, 2
    test rcx, rcx
    jz .neg_f64_remainder

.neg_f64_avx_loop:
    vmovupd ymm0, [rel rsi]
    vxorpd ymm0, ymm0, ymm2
    vmovupd [rel rdi], ymm0
    add rsi, 32
    add rdi, 32
    dec rcx
    jnz .neg_f64_avx_loop

.neg_f64_remainder:
    mov rcx, rbx
    and rcx, 3
    test rcx, rcx
    jz .done

.neg_f64_scalar_loop:
    movsd xmm0, [rel rsi]
    xorpd xmm0, xmm2
    movsd [rel rdi], xmm0
    add rsi, 8
    add rdi, 8
    dec rcx
    jnz .neg_f64_scalar_loop

.done:
    vzeroupper
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; reduce_sum - Sum over specified axis
; Arguments:
;   RDI = Tensor* input
;   RSI = axis (int64_t, -1 for all)
;   RDX = Tensor* output
; Returns:
;   nothing
; =============================================================================
reduce_sum:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; input tensor
    mov r13, rsi                    ; axis
    mov r14, rdx                    ; output tensor
    
    ; For simplicity, handle axis=-1 (sum all elements)
    cmp r13, -1
    je .sum_all
    
    ; TODO: Implement axis-specific reduction
    ; For now, fall through to sum all
    
.sum_all:
    ; Sum all elements
    mov rdi, r12
    call tensor_numel
    mov r15, rax                    ; numel
    
    mov rsi, [r12 + TENSOR_DATA]    ; input data
    mov eax, [r12 + TENSOR_DTYPE]
    
    cmp eax, DT_FLOAT32
    je .sum_all_f32
    cmp eax, DT_FLOAT64
    je .sum_all_f64
    jmp .done

.sum_all_f32:
    vxorps ymm0, ymm0, ymm0         ; accumulator
    
    mov rcx, r15
    shr rcx, 3
    test rcx, rcx
    jz .sum_all_f32_remainder

.sum_all_f32_avx_loop:
    vmovups ymm1, [rel rsi]
    vaddps ymm0, ymm0, ymm1
    add rsi, 32
    dec rcx
    jnz .sum_all_f32_avx_loop
    
    ; Horizontal sum of ymm0
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

.sum_all_f32_remainder:
    mov rcx, r15
    and rcx, 7
    test rcx, rcx
    jz .store_f32

.sum_all_f32_scalar_loop:
    movss xmm1, [rel rsi]
    addss xmm0, xmm1
    add rsi, 4
    dec rcx
    jnz .sum_all_f32_scalar_loop

.store_f32:
    mov rdi, [r14 + TENSOR_DATA]
    movss [rel rdi], xmm0
    jmp .done

.sum_all_f64:
    vxorpd ymm0, ymm0, ymm0
    
    mov rcx, r15
    shr rcx, 2
    test rcx, rcx
    jz .sum_all_f64_remainder

.sum_all_f64_avx_loop:
    vmovupd ymm1, [rel rsi]
    vaddpd ymm0, ymm0, ymm1
    add rsi, 32
    dec rcx
    jnz .sum_all_f64_avx_loop
    
    ; Horizontal sum
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0

.sum_all_f64_remainder:
    mov rcx, r15
    and rcx, 3
    test rcx, rcx
    jz .store_f64

.sum_all_f64_scalar_loop:
    movsd xmm1, [rel rsi]
    addsd xmm0, xmm1
    add rsi, 8
    dec rcx
    jnz .sum_all_f64_scalar_loop

.store_f64:
    mov rdi, [r14 + TENSOR_DATA]
    movsd [rel rdi], xmm0

.done:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; reduce_mean - Mean over specified axis
; Arguments:
;   RDI = Tensor* input
;   RSI = axis (int64_t, -1 for all)
;   RDX = Tensor* output
; =============================================================================
reduce_mean:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    mov r12, rdi                    ; input
    mov r13, rdx                    ; output
    
    ; First compute sum
    call reduce_sum
    
    ; Get element count
    mov rdi, r12
    call tensor_numel
    mov rbx, rax
    
    ; Divide by count
    mov rdi, [r13 + TENSOR_DATA]
    mov eax, [r13 + TENSOR_DTYPE]
    
    cvtsi2sd xmm1, rbx              ; Convert count to double
    
    cmp eax, DT_FLOAT32
    je .mean_f32
    
    ; float64
    movsd xmm0, [rel rdi]
    divsd xmm0, xmm1
    movsd [rel rdi], xmm0
    jmp .done

.mean_f32:
    cvtsd2ss xmm1, xmm1
    movss xmm0, [rel rdi]
    divss xmm0, xmm1
    movss [rel rdi], xmm0

.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; matmul - Matrix multiplication: out = A @ B
; Arguments:
;   RDI = Tensor* out (M x N)
;   RSI = Tensor* A (M x K)
;   RDX = Tensor* B (K x N)
; Assumes row-major layout, 2D tensors
; =============================================================================
matmul:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 72                     ; Local storage
    
    ; Save tensor pointers
    mov [rel rsp], rdi                  ; out
    mov [rsp+8], rsi                ; A
    mov [rsp+16], rdx               ; B
    
    ; Get dimensions
    mov rax, [rsi + TENSOR_SHAPE]
    mov r12, [rel rax]                  ; M
    mov r13, [rax + 8]              ; K
    
    mov rax, [rdx + TENSOR_SHAPE]
    mov r14, [rax + 8]              ; N
    
    ; Save dimensions
    mov [rsp+24], r12               ; M
    mov [rsp+32], r13               ; K
    mov [rsp+40], r14               ; N
    
    ; Get data pointers
    mov rax, [rel rsp]
    mov r15, [rax + TENSOR_DATA]    ; out data
    mov rax, [rsp+8]
    mov rbx, [rax + TENSOR_DATA]    ; A data
    mov rax, [rsp+16]
    mov rcx, [rax + TENSOR_DATA]    ; B data
    mov [rsp+48], rcx               ; Save B data
    
    ; Check dtype
    mov rax, [rsp+8]
    mov eax, [rax + TENSOR_DTYPE]
    mov [rsp+56], eax               ; dtype
    
    cmp eax, DT_FLOAT32
    je .matmul_f32
    cmp eax, DT_FLOAT64
    je .matmul_f64
    jmp .done

.matmul_f32:
    ; Naive O(M*N*K) implementation with some SIMD
    ; For each row i in A (0..M-1)
    xor r8, r8                      ; i = 0
.mm_f32_row_loop:
    cmp r8, r12                     ; i < M
    jge .done
    
    ; For each column j in B (0..N-1)
    xor r9, r9                      ; j = 0
.mm_f32_col_loop:
    cmp r9, r14                     ; j < N
    jge .mm_f32_next_row
    
    ; Compute dot product of A[i,:] and B[:,j]
    vxorps xmm0, xmm0, xmm0         ; accumulator
    
    ; A[i,k] is at A_data + (i*K + k)*4
    ; B[k,j] is at B_data + (k*N + j)*4
    mov rax, r8
    imul rax, r13                   ; i * K
    lea rsi, [rbx + rax*4]          ; &A[i,0]
    
    mov rcx, [rsp+48]               ; B data
    lea rcx, [rcx + r9*4]           ; &B[0,j]
    
    xor r10, r10                    ; k = 0
.mm_f32_dot_loop:
    cmp r10, r13
    jge .mm_f32_store
    
    movss xmm1, [rsi + r10*4]       ; A[i,k]
    ; B[k,j] = B[k*N + j]
    mov rax, r10
    imul rax, r14                   ; k * N
    movss xmm2, [rcx + rax*4]       ; B[k,j]
    mulss xmm1, xmm2
    addss xmm0, xmm1
    
    inc r10
    jmp .mm_f32_dot_loop

.mm_f32_store:
    ; out[i,j] = out[i*N + j]
    mov rax, r8
    imul rax, r14
    add rax, r9
    movss [r15 + rax*4], xmm0
    
    inc r9
    jmp .mm_f32_col_loop

.mm_f32_next_row:
    inc r8
    jmp .mm_f32_row_loop

.matmul_f64:
    ; Similar for float64
    xor r8, r8
.mm_f64_row_loop:
    cmp r8, r12
    jge .done
    
    xor r9, r9
.mm_f64_col_loop:
    cmp r9, r14
    jge .mm_f64_next_row
    
    vxorpd xmm0, xmm0, xmm0
    
    mov rax, r8
    imul rax, r13
    lea rsi, [rbx + rax*8]
    
    mov rcx, [rsp+48]
    lea rcx, [rcx + r9*8]
    
    xor r10, r10
.mm_f64_dot_loop:
    cmp r10, r13
    jge .mm_f64_store
    
    movsd xmm1, [rsi + r10*8]
    mov rax, r10
    imul rax, r14
    movsd xmm2, [rcx + rax*8]
    mulsd xmm1, xmm2
    addsd xmm0, xmm1
    
    inc r10
    jmp .mm_f64_dot_loop

.mm_f64_store:
    mov rax, r8
    imul rax, r14
    add rax, r9
    movsd [r15 + rax*8], xmm0
    
    inc r9
    jmp .mm_f64_col_loop

.mm_f64_next_row:
    inc r8
    jmp .mm_f64_row_loop

.done:
    vzeroupper
    add rsp, 72
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_transpose_2d - Transpose 2D tensor
; Arguments:
;   RDI = Tensor* out (N x M)
;   RSI = Tensor* in (M x N)
; =============================================================================
tensor_transpose_2d:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; in
    
    ; Get dimensions
    mov rax, [r13 + TENSOR_SHAPE]
    mov r14, [rel rax]                  ; M (rows of input)
    mov r15, [rax + 8]              ; N (cols of input)
    
    mov rdi, [r12 + TENSOR_DATA]    ; out data
    mov rsi, [r13 + TENSOR_DATA]    ; in data
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .transpose_f32
    cmp eax, DT_FLOAT64
    je .transpose_f64
    jmp .done

.transpose_f32:
    ; For each element in[i,j], copy to out[j,i]
    xor r8, r8                      ; i = 0
.trans_f32_row_loop:
    cmp r8, r14
    jge .done
    
    xor r9, r9                      ; j = 0
.trans_f32_col_loop:
    cmp r9, r15
    jge .trans_f32_next_row
    
    ; src index = i*N + j
    mov rax, r8
    imul rax, r15
    add rax, r9
    movss xmm0, [rsi + rax*4]
    
    ; dst index = j*M + i
    mov rax, r9
    imul rax, r14
    add rax, r8
    movss [rdi + rax*4], xmm0
    
    inc r9
    jmp .trans_f32_col_loop

.trans_f32_next_row:
    inc r8
    jmp .trans_f32_row_loop

.transpose_f64:
    xor r8, r8
.trans_f64_row_loop:
    cmp r8, r14
    jge .done
    
    xor r9, r9
.trans_f64_col_loop:
    cmp r9, r15
    jge .trans_f64_next_row
    
    mov rax, r8
    imul rax, r15
    add rax, r9
    movsd xmm0, [rsi + rax*8]
    
    mov rax, r9
    imul rax, r14
    add rax, r8
    movsd [rdi + rax*8], xmm0
    
    inc r9
    jmp .trans_f64_col_loop

.trans_f64_next_row:
    inc r8
    jmp .trans_f64_row_loop

.done:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
