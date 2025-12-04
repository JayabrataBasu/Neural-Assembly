; compat.asm - Small symbol aliases and stubs to satisfy caller/export name mismatches
; This file creates thin wrappers so `main.asm` and other modules can link
; quickly. These are compatibility shims and should be replaced by proper
; refactors later (rename or unify symbols).

section .text

; External symbols we forward to
extern tensor_numel
extern backward
extern zero_grad
extern node_relu
extern node_softmax
extern dataset_shuffle_indices

    global tensor_get_size
    global autograd_init
    global autograd_backward
    global autograd_zero_grad
    global relu_forward
    global softmax_forward
    global dataset_load
    global dataset_shuffle
    global mem_init
    global xorshift_seed

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

; dataset_shuffle - No-op (shuffling not implemented)
; The real dataset_shuffle_indices requires an index array which we don't have
dataset_shuffle:
    xor eax, eax
    ret

; mem_init - no-op initializer for memory subsystem
mem_init:
    xor eax, eax
    ret

; xorshift_seed - no-op RNG seed (user should provide real RNG)
xorshift_seed:
    xor eax, eax
    ret
