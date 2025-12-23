; compat.asm - Small symbol aliases and stubs to satisfy caller/export name mismatches
; This file creates thin wrappers so `main.asm` and other modules can link
; quickly. These are compatibility shims and should be replaced by proper
; refactors later (rename or unify symbols).

section .data
    align 8
    rng_state: dq 0x123456789ABCDEF0  ; xorshift64 state (seeded with default)

section .text

; External symbols we forward to
extern tensor_numel
extern backward
extern zero_grad
extern node_relu
extern node_softmax
extern dataset_shuffle_indices
extern dataset_shuffle
extern time

    global tensor_get_size
    global autograd_init
    global autograd_backward
    global autograd_zero_grad
    global relu_forward
    global softmax_forward
    global dataset_load
    global mem_init
    global xorshift_seed
    global xorshift64
    global rand_range

; tensor_get_size -> tensor_numel
tensor_get_size:
    jmp tensor_numel

; autograd_init - noop initializer
autograd_init:
    xor eax, eax
    ret

; autograd_backward -> backward
autograd_backward:
    jmp backward

; autograd_zero_grad -> zero_grad
autograd_zero_grad:
    jmp zero_grad

; relu_forward -> node_relu
relu_forward:
    jmp node_relu

; softmax_forward -> node_softmax
softmax_forward:
    jmp node_softmax

; dataset_load - placeholder
; For now return NULL (no dataset). Caller should handle NULL.
dataset_load:
    xor rax, rax
    ret

; mem_init - no-op initializer for memory subsystem
mem_init:
    xor eax, eax
    ret

; =============================================================================
; xorshift_seed - Seed the xorshift64 RNG
; Arguments:
;   RDI = seed value (if 0, uses current time)
; =============================================================================
xorshift_seed:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jnz .use_seed
    
    ; Get time as seed
    xor edi, edi
    call time wrt ..plt
    mov rdi, rax
    
    ; Make sure seed is not 0
    test rdi, rdi
    jnz .use_seed
    mov rdi, 0x123456789ABCDEF0
    
.use_seed:
    mov [rel rng_state], rdi
    
    pop rbp
    ret

; =============================================================================
; xorshift64 - Generate a 64-bit random number
; Returns:
;   RAX = random 64-bit value
; =============================================================================
xorshift64:
    mov rax, [rel rng_state]
    
    ; xorshift64 algorithm
    mov rcx, rax
    shl rcx, 13
    xor rax, rcx
    
    mov rcx, rax
    shr rcx, 7
    xor rax, rcx
    
    mov rcx, rax
    shl rcx, 17
    xor rax, rcx
    
    mov [rel rng_state], rax
    ret

; =============================================================================
; rand_range - Generate random number in range [0, max)
; Arguments:
;   RDI = max (exclusive upper bound)
; Returns:
;   RAX = random value in [0, max)
; =============================================================================
rand_range:
    push rbp
    mov rbp, rsp
    push rbx
    
    mov rbx, rdi                ; save max
    
    call xorshift64             ; get random 64-bit value
    
    ; rax % rbx (unsigned)
    xor rdx, rdx
    div rbx                     ; rdx = rax % rbx
    mov rax, rdx
    
    pop rbx
    pop rbp
    ret
