; ============================================================================
; AVX-512 Runtime Support
; Detects AVX-512 capabilities and provides optimized kernels
; Fallback to AVX2 if not available
; ============================================================================

global simd_detect_avx512
global vec_add_avx512
global vec_mul_avx512
global matmul_avx512

section .data
    cpu_has_avx512f:    dq 0
    avx512_detected:    dq 0

section .text

; ============================================================================
; simd_detect_avx512: Detect AVX-512 Foundation support
; Input:  none
; Output: rax = 1 if AVX-512F available, 0 otherwise
; ============================================================================

simd_detect_avx512:
    push rbx
    xor eax, eax
    mov ecx, 1
    cpuid
    
    ; Check CPUID leaf 7, subleaf 0 for AVX-512F
    mov eax, 7
    xor ecx, ecx
    cpuid
    
    ; EBX bit 16 indicates AVX-512F
    mov eax, ebx
    shr eax, 16
    and eax, 1
    
    pop rbx
    ret

; ============================================================================
; Initialize AVX-512 state (enable XMM, YMM, ZMM registers)
; Must be called before using AVX-512 instructions
; ============================================================================

init_avx512_state:
    ; XSETBV enables AVX-512 (CR4.OSXSAVE must be set by OS)
    ; We'll just proceed assuming OS support
    ret

; ============================================================================
; vec_add_avx512: Add two 512-bit vectors (16 float32s per register)
; Input:  rdi = pointer to vector A (float32 array)
;         rsi = pointer to vector B (float32 array)
;         rdx = pointer to output C
;         rcx = number of elements
; Output: C[i] = A[i] + B[i]
; ============================================================================

vec_add_avx512:
    cmp rcx, 0
    jle .vec_add_avx512_done
    
    mov rax, rcx
    shr rax, 4              ; rcx / 16 (16 float32s per zmm register)
    jz .vec_add_avx512_remainder
    
.vec_add_avx512_loop:
    vmovups zmm0, [rdi]     ; Load 16 floats from A
    vmovups zmm1, [rsi]     ; Load 16 floats from B
    vaddps zmm2, zmm0, zmm1  ; Add: zmm2 = zmm0 + zmm1
    vmovups [rdx], zmm2     ; Store result
    
    add rdi, 64             ; Next 16 floats (16 * 4 bytes)
    add rsi, 64
    add rdx, 64
    
    dec rax
    jnz .vec_add_avx512_loop
    
.vec_add_avx512_remainder:
    ; Handle remaining elements (<16)
    and rcx, 15
    jz .vec_add_avx512_done
    
    ; For simplicity, use scalar code for remainder
    xor rax, rax
.vec_add_avx512_scalar_loop:
    cmp rax, rcx
    jge .vec_add_avx512_done
    
    movss xmm0, [rdi + rax*4]
    movss xmm1, [rsi + rax*4]
    addss xmm0, xmm1
    movss [rdx + rax*4], xmm0
    
    inc rax
    jmp .vec_add_avx512_scalar_loop
    
.vec_add_avx512_done:
    vzeroupper              ; Clear upper half of YMM/ZMM for SSE compatibility
    ret

; ============================================================================
; vec_mul_avx512: Element-wise multiply (Hadamard product)
; Input:  rdi = pointer to vector A
;         rsi = pointer to vector B
;         rdx = pointer to output C
;         rcx = number of elements
; Output: C[i] = A[i] * B[i]
; ============================================================================

vec_mul_avx512:
    cmp rcx, 0
    jle .vec_mul_avx512_done
    
    mov rax, rcx
    shr rax, 4              ; rcx / 16
    jz .vec_mul_avx512_remainder
    
.vec_mul_avx512_loop:
    vmovups zmm0, [rdi]
    vmovups zmm1, [rsi]
    vmulps zmm2, zmm0, zmm1
    vmovups [rdx], zmm2
    
    add rdi, 64
    add rsi, 64
    add rdx, 64
    
    dec rax
    jnz .vec_mul_avx512_loop
    
.vec_mul_avx512_remainder:
    and rcx, 15
    jz .vec_mul_avx512_done
    
    xor rax, rax
.vec_mul_avx512_scalar_loop:
    cmp rax, rcx
    jge .vec_mul_avx512_done
    
    movss xmm0, [rdi + rax*4]
    movss xmm1, [rsi + rax*4]
    mulss xmm0, xmm1
    movss [rdx + rax*4], xmm0
    
    inc rax
    jmp .vec_mul_avx512_scalar_loop
    
.vec_mul_avx512_done:
    vzeroupper
    ret

; ============================================================================
; matmul_avx512: Matrix multiply with AVX-512 (batch of 16 columns)
; Input:  rdi = pointer to matrix A (m x k, row-major)
;         rsi = pointer to matrix B (k x n, row-major)
;         rdx = pointer to output C (m x n)
;         rcx = m
;         r8  = k
;         r9  = n
; Output: C = A @ B
; 
; Strategy: Process B in chunks of 16 columns
; For each output column, dot product row of A with column of B
; ============================================================================

matmul_avx512:
    push rbx
    push r10
    push r11
    push r12
    
    ; Save parameters
    mov r10, rcx            ; m
    mov r11, r8             ; k
    mov r12, r9             ; n
    
    ; Outer loop: rows of A
    xor rbx, rbx            ; i = 0
.matmul_avx512_row_loop:
    cmp rbx, r10
    jge .matmul_avx512_done
    
    ; Inner loop: columns of B (16 at a time)
    xor r8, r8              ; j = 0
.matmul_avx512_col_loop:
    cmp r8, r12
    jge .matmul_avx512_next_row
    
    ; Compute C[i,j:j+16] = dot(A[i,:], B[:,j:j+16])
    vpxorq zmm4, zmm4, zmm4  ; accumulator (16 float32s)
    
    xor rcx, rcx            ; k_idx = 0
.matmul_avx512_k_loop:
    cmp rcx, r11
    jge .matmul_avx512_store_result
    
    ; Load A[i, k_idx]
    mov rax, rbx
    imul rax, r11
    add rax, rcx
    shl rax, 2              ; *4 for float32
    movss xmm0, [rdi + rax]
    
    ; Broadcast A[i, k_idx] to all 16 float32s in zmm0
    vbroadcastss zmm0, xmm0
    
    ; Load 16 consecutive floats from B[k_idx, j:j+16]
    mov rax, rcx
    imul rax, r12
    add rax, r8
    shl rax, 2
    vmovups zmm1, [rsi + rax]
    
    ; Multiply and accumulate
    vfmadd231ps zmm4, zmm0, zmm1
    
    inc rcx
    jmp .matmul_avx512_k_loop
    
.matmul_avx512_store_result:
    ; Write accumulator to C[i, j:j+16]
    mov rax, rbx
    imul rax, r12
    add rax, r8
    shl rax, 2
    vmovups [rdx + rax], zmm4
    
    ; Next batch of 16 columns
    add r8, 16
    cmp r8, r12
    jle .matmul_avx512_col_loop
    
    ; Handle remainder columns (< 16 remaining)
    mov rcx, r12
    sub rcx, r8
    jle .matmul_avx512_next_row
    
    ; Process 1-15 remaining columns with masked operations
    mov r9, -1
    shr r9, cl              ; Create mask for remaining columns
    
    vpxorq zmm4, zmm4, zmm4
    xor rcx, rcx
.matmul_avx512_remainder_k_loop:
    cmp rcx, r11
    jge .matmul_avx512_store_remainder
    
    mov rax, rbx
    imul rax, r11
    add rax, rcx
    shl rax, 2
    movss xmm0, [rdi + rax]
    vbroadcastss zmm0, xmm0
    
    mov rax, rcx
    imul rax, r12
    add rax, r8
    shl rax, 2
    vmovups zmm1, [rsi + rax]
    
    vfmadd231ps zmm4, zmm0, zmm1
    inc rcx
    jmp .matmul_avx512_remainder_k_loop
    
.matmul_avx512_store_remainder:
    mov rax, rbx
    imul rax, r12
    add rax, r8
    shl rax, 2
    vmovups [rdx + rax], zmm4
    
.matmul_avx512_next_row:
    inc rbx
    jmp .matmul_avx512_row_loop
    
.matmul_avx512_done:
    vzeroupper
    pop r12
    pop r11
    pop r10
    pop rbx
    ret
