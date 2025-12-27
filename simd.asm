; =============================================================================
; simd.asm - SIMD Detection and Optimized Kernels
; =============================================================================
; Runtime CPU feature detection, AVX-512/AVX2/SSE dispatch
; =============================================================================

section .data
    align 8
    
    ; Feature flags (set at runtime)
    has_sse:        dd 0
    has_sse2:       dd 0
    has_sse3:       dd 0
    has_ssse3:      dd 0
    has_sse41:      dd 0
    has_sse42:      dd 0
    has_avx:        dd 0
    has_avx2:       dd 0
    has_fma:        dd 0
    has_avx512f:    dd 0
    has_avx512dq:   dd 0
    has_avx512vl:   dd 0
    
    ; Feature detection done flag
    features_detected:  dd 0
    
    ; Messages
    msg_sse:        db "SSE", 0
    msg_sse2:       db "SSE2", 0
    msg_avx:        db "AVX", 0
    msg_avx2:       db "AVX2", 0
    msg_avx512:     db "AVX-512", 0
    msg_fma:        db "FMA", 0
    msg_detected:   db " detected", 10, 0
    msg_using:      db "[rel SIMD] Using: ", 0
    msg_newline:    db 10, 0

section .bss
    align 64
    ; Scratch space for SIMD operations (64-byte aligned for AVX-512)
    simd_scratch:   resb 512

section .text

extern printf

; Export SIMD functions
global detect_cpu_features
global get_simd_level
global simd_add_f32
global simd_sub_f32
global simd_mul_f32
global simd_div_f32
global simd_fma_f32
global simd_dot_f32
global simd_sum_f32
global simd_max_f32

; =============================================================================
; detect_cpu_features - Detect CPU SIMD capabilities
; Returns:
;   EAX = highest SIMD level (0=scalar, 1=SSE, 2=AVX, 3=AVX2, 4=AVX-512)
; =============================================================================
detect_cpu_features:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 8
    
    ; Check if already detected
    cmp dword [rel features_detected], 1
    je .return_level
    
    ; CPUID check for SSE/SSE2 (function 1)
    mov eax, 1
    xor ecx, ecx
    cpuid
    
    ; EDX bit 25 = SSE, bit 26 = SSE2
    bt edx, 25
    setc byte [rel has_sse]
    bt edx, 26
    setc byte [rel has_sse2]
    
    ; ECX bit 0 = SSE3, bit 9 = SSSE3, bit 19 = SSE4.1, bit 20 = SSE4.2
    bt ecx, 0
    setc byte [rel has_sse3]
    bt ecx, 9
    setc byte [rel has_ssse3]
    bt ecx, 19
    setc byte [rel has_sse41]
    bt ecx, 20
    setc byte [rel has_sse42]
    
    ; ECX bit 28 = AVX, bit 12 = FMA
    bt ecx, 28
    setc byte [rel has_avx]
    bt ecx, 12
    setc byte [rel has_fma]
    
    ; Check for AVX2 (CPUID function 7, subleaf 0)
    mov eax, 7
    xor ecx, ecx
    cpuid
    
    ; EBX bit 5 = AVX2
    bt ebx, 5
    setc byte [rel has_avx2]
    
    ; EBX bit 16 = AVX-512F, bit 17 = AVX-512DQ
    bt ebx, 16
    setc byte [rel has_avx512f]
    bt ebx, 17
    setc byte [rel has_avx512dq]
    
    ; EBX bit 31 = AVX-512VL
    bt ebx, 31
    setc byte [rel has_avx512vl]
    
    mov dword [rel features_detected], 1
    
.return_level:
    ; Determine highest level
    xor eax, eax                ; 0 = scalar
    
    cmp byte [rel has_sse2], 0
    je .done
    mov eax, 1                  ; SSE
    
    cmp byte [rel has_avx], 0
    je .done
    mov eax, 2                  ; AVX
    
    cmp byte [rel has_avx2], 0
    je .done
    mov eax, 3                  ; AVX2
    
    cmp byte [rel has_avx512f], 0
    je .done
    mov eax, 4                  ; AVX-512
    
.done:
    add rsp, 8
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; get_simd_level - Get current SIMD level
; Returns:
;   EAX = SIMD level (0-4)
; =============================================================================
get_simd_level:
    push rbp
    mov rbp, rsp
    
    cmp dword [rel features_detected], 0
    jne .get_level
    call detect_cpu_features
    
.get_level:
    xor eax, eax
    cmp byte [rel has_sse2], 0
    je .done
    inc eax
    cmp byte [rel has_avx], 0
    je .done
    inc eax
    cmp byte [rel has_avx2], 0
    je .done
    inc eax
    cmp byte [rel has_avx512f], 0
    je .done
    inc eax
    
.done:
    pop rbp
    ret

; =============================================================================
; simd_add_f32 - Optimized float32 addition with best available SIMD
; Arguments:
;   RDI = float* dst
;   RSI = float* a
;   RDX = float* b
;   RCX = count
; =============================================================================
simd_add_f32:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov r12, rdi                ; dst
    mov r13, rsi                ; a
    mov r14, rdx                ; b
    mov r15, rcx                ; count
    
    ; Detect SIMD level
    call get_simd_level
    
    cmp eax, 4
    je .avx512_add
    cmp eax, 3
    jge .avx2_add
    cmp eax, 2
    jge .avx_add
    jmp .sse_add
    
.avx512_add:
    ; Process 16 floats at a time
    mov rcx, r15
    shr rcx, 4                  ; count / 16
    test rcx, rcx
    jz .avx512_remainder
    
.avx512_loop:
    vmovups zmm0, [rel r13]
    vmovups zmm1, [rel r14]
    vaddps zmm0, zmm0, zmm1
    vmovups [rel r12], zmm0
    add r13, 64
    add r14, 64
    add r12, 64
    dec rcx
    jnz .avx512_loop
    
.avx512_remainder:
    mov rcx, r15
    and rcx, 15
    jmp .scalar_remainder
    
.avx2_add:
.avx_add:
    ; Process 8 floats at a time
    mov rcx, r15
    shr rcx, 3
    test rcx, rcx
    jz .avx_remainder
    
.avx_loop:
    vmovups ymm0, [rel r13]
    vmovups ymm1, [rel r14]
    vaddps ymm0, ymm0, ymm1
    vmovups [rel r12], ymm0
    add r13, 32
    add r14, 32
    add r12, 32
    dec rcx
    jnz .avx_loop
    
.avx_remainder:
    mov rcx, r15
    and rcx, 7
    jmp .scalar_remainder
    
.sse_add:
    ; Process 4 floats at a time
    mov rcx, r15
    shr rcx, 2
    test rcx, rcx
    jz .sse_remainder
    
.sse_loop:
    movups xmm0, [rel r13]
    movups xmm1, [rel r14]
    addps xmm0, xmm1
    movups [rel r12], xmm0
    add r13, 16
    add r14, 16
    add r12, 16
    dec rcx
    jnz .sse_loop
    
.sse_remainder:
    mov rcx, r15
    and rcx, 3
    
.scalar_remainder:
    test rcx, rcx
    jz .add_done
    
.scalar_loop:
    movss xmm0, [rel r13]
    addss xmm0, [rel r14]
    movss [rel r12], xmm0
    add r13, 4
    add r14, 4
    add r12, 4
    dec rcx
    jnz .scalar_loop
    
.add_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; simd_mul_f32 - Optimized float32 multiplication with best available SIMD
; Arguments:
;   RDI = float* dst
;   RSI = float* a
;   RDX = float* b
;   RCX = count
; =============================================================================
simd_mul_f32:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    
    call get_simd_level
    
    cmp eax, 4
    je .avx512_mul
    cmp eax, 2
    jge .avx_mul
    jmp .sse_mul
    
.avx512_mul:
    mov rcx, r15
    shr rcx, 4
    test rcx, rcx
    jz .avx512_mul_rem
    
.avx512_mul_loop:
    vmovups zmm0, [rel r13]
    vmovups zmm1, [rel r14]
    vmulps zmm0, zmm0, zmm1
    vmovups [rel r12], zmm0
    add r13, 64
    add r14, 64
    add r12, 64
    dec rcx
    jnz .avx512_mul_loop
    
.avx512_mul_rem:
    mov rcx, r15
    and rcx, 15
    jmp .mul_scalar_rem
    
.avx_mul:
    mov rcx, r15
    shr rcx, 3
    test rcx, rcx
    jz .avx_mul_rem
    
.avx_mul_loop:
    vmovups ymm0, [rel r13]
    vmovups ymm1, [rel r14]
    vmulps ymm0, ymm0, ymm1
    vmovups [rel r12], ymm0
    add r13, 32
    add r14, 32
    add r12, 32
    dec rcx
    jnz .avx_mul_loop
    
.avx_mul_rem:
    mov rcx, r15
    and rcx, 7
    jmp .mul_scalar_rem
    
.sse_mul:
    mov rcx, r15
    shr rcx, 2
    test rcx, rcx
    jz .sse_mul_rem
    
.sse_mul_loop:
    movups xmm0, [rel r13]
    movups xmm1, [rel r14]
    mulps xmm0, xmm1
    movups [rel r12], xmm0
    add r13, 16
    add r14, 16
    add r12, 16
    dec rcx
    jnz .sse_mul_loop
    
.sse_mul_rem:
    mov rcx, r15
    and rcx, 3
    
.mul_scalar_rem:
    test rcx, rcx
    jz .mul_done
    
.mul_scalar_loop:
    movss xmm0, [rel r13]
    mulss xmm0, [rel r14]
    movss [rel r12], xmm0
    add r13, 4
    add r14, 4
    add r12, 4
    dec rcx
    jnz .mul_scalar_loop
    
.mul_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; simd_fma_f32 - Fused multiply-add: dst = a * b + c
; Arguments:
;   RDI = float* dst
;   RSI = float* a
;   RDX = float* b
;   RCX = float* c
;   R8  = count
; =============================================================================
simd_fma_f32:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi                ; dst
    mov r13, rsi                ; a
    mov r14, rdx                ; b
    mov r15, rcx                ; c
    mov rbx, r8                 ; count
    
    ; Check for FMA support
    cmp byte [rel has_fma], 0
    je .no_fma
    
    call get_simd_level
    cmp eax, 4
    je .avx512_fma
    cmp eax, 2
    jge .avx_fma
    jmp .no_fma
    
.avx512_fma:
    mov rcx, rbx
    shr rcx, 4
    test rcx, rcx
    jz .avx512_fma_rem
    
.avx512_fma_loop:
    vmovups zmm0, [rel r13]         ; a
    vmovups zmm1, [rel r14]         ; b
    vmovups zmm2, [rel r15]         ; c
    vfmadd213ps zmm0, zmm1, zmm2 ; a = a*b + c
    vmovups [rel r12], zmm0
    add r13, 64
    add r14, 64
    add r15, 64
    add r12, 64
    dec rcx
    jnz .avx512_fma_loop
    
.avx512_fma_rem:
    mov rcx, rbx
    and rcx, 15
    jmp .fma_scalar
    
.avx_fma:
    mov rcx, rbx
    shr rcx, 3
    test rcx, rcx
    jz .avx_fma_rem
    
.avx_fma_loop:
    vmovups ymm0, [rel r13]
    vmovups ymm1, [rel r14]
    vmovups ymm2, [rel r15]
    vfmadd213ps ymm0, ymm1, ymm2
    vmovups [rel r12], ymm0
    add r13, 32
    add r14, 32
    add r15, 32
    add r12, 32
    dec rcx
    jnz .avx_fma_loop
    
.avx_fma_rem:
    mov rcx, rbx
    and rcx, 7
    jmp .fma_scalar
    
.no_fma:
    ; Fallback: mul then add
    mov rcx, rbx
    
.fma_scalar:
    test rcx, rcx
    jz .fma_done
    
.fma_scalar_loop:
    movss xmm0, [rel r13]
    mulss xmm0, [rel r14]
    addss xmm0, [rel r15]
    movss [rel r12], xmm0
    add r13, 4
    add r14, 4
    add r15, 4
    add r12, 4
    dec rcx
    jnz .fma_scalar_loop
    
.fma_done:
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
; simd_dot_f32 - Dot product of two float32 arrays
; Arguments:
;   RDI = float* a
;   RSI = float* b
;   RDX = count
; Returns:
;   XMM0 = dot product (float)
; =============================================================================
simd_dot_f32:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    vxorps ymm0, ymm0, ymm0     ; accumulator
    
    call get_simd_level
    
    cmp eax, 4
    je .avx512_dot
    cmp eax, 2
    jge .avx_dot
    jmp .sse_dot
    
.avx512_dot:
    vxorps zmm0, zmm0, zmm0
    mov rcx, r14
    shr rcx, 4
    test rcx, rcx
    jz .avx512_dot_rem
    
.avx512_dot_loop:
    vmovups zmm1, [rel r12]
    vmovups zmm2, [rel r13]
    vfmadd231ps zmm0, zmm1, zmm2
    add r12, 64
    add r13, 64
    dec rcx
    jnz .avx512_dot_loop
    
    ; Reduce zmm0 to scalar
    vextractf32x8 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    
.avx512_dot_rem:
    mov rcx, r14
    and rcx, 15
    jmp .dot_scalar
    
.avx_dot:
    vxorps ymm0, ymm0, ymm0
    mov rcx, r14
    shr rcx, 3
    test rcx, rcx
    jz .avx_dot_rem
    
.avx_dot_loop:
    vmovups ymm1, [rel r12]
    vmovups ymm2, [rel r13]
    vfmadd231ps ymm0, ymm1, ymm2
    add r12, 32
    add r13, 32
    dec rcx
    jnz .avx_dot_loop
    
    ; Reduce ymm0
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    
.avx_dot_rem:
    mov rcx, r14
    and rcx, 7
    jmp .dot_scalar
    
.sse_dot:
    xorps xmm0, xmm0
    mov rcx, r14
    shr rcx, 2
    test rcx, rcx
    jz .sse_dot_rem
    
.sse_dot_loop:
    movups xmm1, [rel r12]
    movups xmm2, [rel r13]
    mulps xmm1, xmm2
    addps xmm0, xmm1
    add r12, 16
    add r13, 16
    dec rcx
    jnz .sse_dot_loop
    
    ; Reduce xmm0
    haddps xmm0, xmm0
    haddps xmm0, xmm0
    
.sse_dot_rem:
    mov rcx, r14
    and rcx, 3
    
.dot_scalar:
    test rcx, rcx
    jz .dot_done
    
.dot_scalar_loop:
    movss xmm1, [rel r12]
    mulss xmm1, [rel r13]
    addss xmm0, xmm1
    add r12, 4
    add r13, 4
    dec rcx
    jnz .dot_scalar_loop
    
.dot_done:
    vzeroupper
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; simd_sum_f32 - Sum all elements in float32 array
; Arguments:
;   RDI = float* arr
;   RSI = count
; Returns:
;   XMM0 = sum (float)
; =============================================================================
simd_sum_f32:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    
    mov r12, rdi
    mov r13, rsi
    
    call get_simd_level
    
    cmp eax, 4
    je .avx512_sum
    cmp eax, 2
    jge .avx_sum
    jmp .sse_sum
    
.avx512_sum:
    vxorps zmm0, zmm0, zmm0
    mov rcx, r13
    shr rcx, 4
    test rcx, rcx
    jz .avx512_sum_rem
    
.avx512_sum_loop:
    vaddps zmm0, zmm0, [rel r12]
    add r12, 64
    dec rcx
    jnz .avx512_sum_loop
    
    vextractf32x8 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    
.avx512_sum_rem:
    mov rcx, r13
    and rcx, 15
    jmp .sum_scalar
    
.avx_sum:
    vxorps ymm0, ymm0, ymm0
    mov rcx, r13
    shr rcx, 3
    test rcx, rcx
    jz .avx_sum_rem
    
.avx_sum_loop:
    vaddps ymm0, ymm0, [rel r12]
    add r12, 32
    dec rcx
    jnz .avx_sum_loop
    
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    
.avx_sum_rem:
    mov rcx, r13
    and rcx, 7
    jmp .sum_scalar
    
.sse_sum:
    xorps xmm0, xmm0
    mov rcx, r13
    shr rcx, 2
    test rcx, rcx
    jz .sse_sum_rem
    
.sse_sum_loop:
    addps xmm0, [rel r12]
    add r12, 16
    dec rcx
    jnz .sse_sum_loop
    
    haddps xmm0, xmm0
    haddps xmm0, xmm0
    
.sse_sum_rem:
    mov rcx, r13
    and rcx, 3
    
.sum_scalar:
    test rcx, rcx
    jz .sum_done
    
.sum_scalar_loop:
    addss xmm0, [rel r12]
    add r12, 4
    dec rcx
    jnz .sum_scalar_loop
    
.sum_done:
    vzeroupper
    pop r13
    pop r12
    pop rbp
    ret
; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
