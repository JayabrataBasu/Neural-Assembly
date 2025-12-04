; =============================================================================
; optimizers.asm - Optimization Algorithms
; =============================================================================
; SGD, Adam optimizers
; =============================================================================

; Optimizer struct layout (64 bytes):
; Offset  Size    Field
; 0       4       n_params     (uint32_t)
; 4       4       padding
; 8       8       params       (Tensor**) - parameter tensors
; 16      8       param_nodes  (Node**) - parameter nodes (grads from node->grad)
; 24      8       step_fn      (void (*)(Optimizer*))
; 32      8       zero_grad_fn (void (*)(Optimizer*))
; 40      8       state        (void*) - optimizer-specific state
; 48      16      reserved

%define OPT_SIZE            64
%define OPT_N_PARAMS        0
%define OPT_PARAMS          8
%define OPT_PARAM_NODES     16
%define OPT_STEP_FN         24
%define OPT_ZERO_GRAD_FN    32
%define OPT_STATE           40

; Node offset for grad
%define NODE_GRAD           8

; SGD State struct:
; Offset  Size    Field
; 0       8       lr           (double)
; 8       8       momentum     (double)
; 16      8       velocities   (Tensor**) - velocity tensors for momentum

; Adam State struct:
; Offset  Size    Field
; 0       8       lr           (double)
; 8       8       beta1        (double)
; 16      8       beta2        (double)
; 24      8       eps          (double)
; 32      8       t            (uint64_t) - time step
; 40      8       m            (Tensor**) - first moment tensors
; 48      8       v            (Tensor**) - second moment tensors

%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_DTYPE        32

%define DT_FLOAT32          0
%define DT_FLOAT64          1

section .data
    align 8
    one_f64:            dq 1.0

section .bss
    align 32

section .text

; External functions
extern mem_alloc
extern mem_alloc_aligned
extern mem_free
extern mem_zero
extern tensor_create
extern tensor_zeros
extern tensor_copy
extern tensor_free
extern tensor_numel
extern tensor_data_size
extern ew_add
extern ew_sub
extern ew_mul
extern ew_scalar_mul
extern pow
extern sqrt

; Export optimizer functions
global sgd_create
global sgd_step
global sgd_zero_grad
global adam_create
global adam_step
global adam_zero_grad
global optimizer_free

; =============================================================================
; sgd_create - Create SGD optimizer
; Arguments:
;   RDI = Tensor** params (array of parameter tensors)
;   RSI = Node** param_nodes (array of parameter nodes - grads from node->grad)
;   RDX = n_params (uint32_t)
;   XMM0 = lr (double)
;   XMM1 = momentum (double)
; Returns:
;   RAX = Optimizer*
; =============================================================================
sgd_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; params
    mov r13, rsi                    ; param_nodes
    mov r14d, edx                   ; n_params
    movsd [rsp], xmm0               ; lr
    movsd [rsp+8], xmm1             ; momentum
    
    ; Allocate optimizer struct
    mov rdi, OPT_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed
    mov r15, rax
    
    ; Initialize optimizer
    mov [r15 + OPT_N_PARAMS], r14d
    mov [r15 + OPT_PARAMS], r12
    mov [r15 + OPT_PARAM_NODES], r13
    
    lea rax, [rel sgd_step]
    mov [r15 + OPT_STEP_FN], rax
    lea rax, [rel sgd_zero_grad]
    mov [r15 + OPT_ZERO_GRAD_FN], rax
    
    ; Allocate SGD state (24 bytes + velocity array)
    mov rdi, 24
    call mem_alloc
    mov [r15 + OPT_STATE], rax
    mov rbx, rax
    
    ; Copy hyperparameters
    movsd xmm0, [rsp]
    movsd [rbx], xmm0               ; lr
    movsd xmm0, [rsp+8]
    movsd [rbx+8], xmm0             ; momentum
    
    ; Check if momentum is used
    vxorpd xmm1, xmm1, xmm1
    ucomisd xmm0, xmm1
    je .no_momentum
    
    ; Allocate velocity tensors array
    mov eax, r14d
    shl eax, 3                      ; n_params * 8
    mov edi, eax
    call mem_alloc
    mov [rbx+16], rax               ; velocities array
    mov r13, rax
    
    ; Create velocity tensors (zeros, same shape as params)
    xor ecx, ecx
.create_velocities:
    cmp ecx, r14d
    jge .done                       ; done creating velocities, jump to .done
    
    push rcx
    mov rax, [r12 + rcx*8]          ; params[i]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_zeros
    pop rcx
    
    mov [r13 + rcx*8], rax
    inc ecx
    jmp .create_velocities

.no_momentum:
    mov qword [rbx+16], 0

.done:
    mov rax, r15
    
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; sgd_step - Perform one optimization step
; Arguments:
;   RDI = Optimizer* opt
; =============================================================================
sgd_step:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; optimizer
    
    mov r13d, [r12 + OPT_N_PARAMS]
    mov r14, [r12 + OPT_PARAMS]
    mov r15, [r12 + OPT_PARAM_NODES] ; param_nodes array
    mov rbx, [r12 + OPT_STATE]
    
    movsd xmm4, [rbx]               ; lr
    movsd xmm5, [rbx+8]             ; momentum
    mov rax, [rbx+16]               ; velocities (may be null)
    mov [rsp], rax
    
    ; Check if using momentum
    vxorpd xmm0, xmm0, xmm0
    ucomisd xmm5, xmm0
    je .no_momentum_step
    
    ; With momentum: v = momentum * v + grad; param -= lr * v
    xor ecx, ecx
.momentum_loop:
    cmp ecx, r13d
    jge .done
    
    mov [rsp+8], ecx                ; save index
    
    mov rax, [r14 + rcx*8]          ; param tensor
    ; Get grad from param_nodes[i]->grad (NODE_GRAD = 8)
    mov rsi, [r15 + rcx*8]          ; param_node
    test rsi, rsi
    jz .next_param                  ; skip if null node
    mov rsi, [rsi + NODE_GRAD]      ; node->grad (tensor)
    test rsi, rsi
    jz .next_param                  ; skip if null grad
    mov [rsp+48], rsi               ; save grad tensor
    mov r8, [rsp]
    mov rdx, [r8 + rcx*8]           ; velocity tensor
    
    mov rdi, [rax + TENSOR_DATA]
    mov r8, [rsi + TENSOR_DATA]
    mov r9, [rdx + TENSOR_DATA]
    
    ; Get numel
    push rdi
    push r8
    push r9
    mov rdi, rax
    call tensor_numel
    mov r10, rax
    pop r9
    pop r8
    pop rdi
    
    mov rax, [rsp+48]               ; reload grad tensor
    mov eax, [rax + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .momentum_f32
    
    ; float64
    xor r11, r11
.momentum_f64_loop:
    cmp r11, r10
    jge .next_param
    
    ; v = momentum * v + grad
    movsd xmm0, [r9 + r11*8]        ; v
    mulsd xmm0, xmm5                ; momentum * v
    addsd xmm0, [r8 + r11*8]        ; + grad
    movsd [r9 + r11*8], xmm0        ; store v
    
    ; param -= lr * v
    mulsd xmm0, xmm4                ; lr * v
    movsd xmm1, [rdi + r11*8]       ; param
    subsd xmm1, xmm0
    movsd [rdi + r11*8], xmm1
    
    inc r11
    jmp .momentum_f64_loop

.momentum_f32:
    cvtsd2ss xmm4, xmm4             ; lr as float
    cvtsd2ss xmm5, xmm5             ; momentum as float
    
    xor r11, r11
.momentum_f32_loop:
    cmp r11, r10
    jge .next_param_restore
    
    movss xmm0, [r9 + r11*4]
    mulss xmm0, xmm5
    addss xmm0, [r8 + r11*4]
    movss [r9 + r11*4], xmm0
    
    mulss xmm0, xmm4
    movss xmm1, [rdi + r11*4]
    subss xmm1, xmm0
    movss [rdi + r11*4], xmm1
    
    inc r11
    jmp .momentum_f32_loop

.next_param_restore:
    ; Restore lr and momentum to double
    movsd xmm4, [rbx]
    movsd xmm5, [rbx+8]

.next_param:
    mov ecx, [rsp+8]
    inc ecx
    jmp .momentum_loop

.no_momentum_step:
    ; Without momentum: param -= lr * grad
    xor ecx, ecx
.simple_loop:
    cmp ecx, r13d
    jge .done
    
    mov [rsp+8], ecx
    
    mov rax, [r14 + rcx*8]          ; param tensor
    ; Get grad from param_nodes[i]->grad
    mov rsi, [r15 + rcx*8]          ; param_node
    test rsi, rsi
    jz .simple_next                 ; skip if null node
    mov rsi, [rsi + NODE_GRAD]      ; node->grad
    test rsi, rsi
    jz .simple_next                 ; skip if null grad
    mov [rsp+48], rsi               ; save grad tensor
    
    mov rdi, [rax + TENSOR_DATA]
    mov r8, [rsi + TENSOR_DATA]
    
    push rdi
    push r8
    mov rdi, rax
    call tensor_numel
    mov r10, rax
    pop r8
    pop rdi
    
    mov rax, [rsp+48]               ; reload grad tensor
    mov eax, [rax + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .simple_f32
    
    ; float64
    xor r11, r11
.simple_f64_loop:
    cmp r11, r10
    jge .simple_next
    
    movsd xmm0, [r8 + r11*8]        ; grad
    mulsd xmm0, xmm4                ; lr * grad
    movsd xmm1, [rdi + r11*8]       ; param
    subsd xmm1, xmm0
    movsd [rdi + r11*8], xmm1
    
    inc r11
    jmp .simple_f64_loop

.simple_f32:
    cvtsd2ss xmm4, xmm4
    
    xor r11, r11
.simple_f32_loop:
    cmp r11, r10
    jge .simple_next_restore
    
    movss xmm0, [r8 + r11*4]
    mulss xmm0, xmm4
    movss xmm1, [rdi + r11*4]
    subss xmm1, xmm0
    movss [rdi + r11*4], xmm1
    
    inc r11
    jmp .simple_f32_loop

.simple_next_restore:
    movsd xmm4, [rbx]

.simple_next:
    mov ecx, [rsp+8]
    inc ecx
    jmp .simple_loop

.done:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; sgd_zero_grad - Zero all gradients
; Arguments:
;   RDI = Optimizer* opt
; =============================================================================
sgd_zero_grad:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    mov r12, rdi
    mov r13d, [r12 + OPT_N_PARAMS]
    mov rbx, [r12 + OPT_PARAM_NODES]
    
    xor ecx, ecx
.zero_loop:
    cmp ecx, r13d
    jge .done
    
    push rcx
    mov rax, [rbx + rcx*8]          ; param_node
    test rax, rax
    jz .next
    mov rax, [rax + NODE_GRAD]      ; node->grad tensor
    test rax, rax
    jz .next
    
    push rax
    mov rdi, rax
    call tensor_data_size
    mov rsi, rax
    pop rax
    mov rdi, [rax + TENSOR_DATA]
    call mem_zero
    
.next:
    pop rcx
    inc ecx
    jmp .zero_loop

.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; adam_create - Create Adam optimizer
; Arguments:
;   RDI = Tensor** params
;   RSI = Node** param_nodes
;   RDX = n_params (uint32_t)
;   XMM0 = lr (double)
;   XMM1 = beta1 (double)
;   XMM2 = beta2 (double)
;   XMM3 = eps (double)
; Returns:
;   RAX = Optimizer*
; =============================================================================
adam_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 72
    
    mov r12, rdi                    ; params
    mov r13, rsi                    ; param_nodes
    mov r14d, edx                   ; n_params
    movsd [rsp], xmm0               ; lr
    movsd [rsp+8], xmm1             ; beta1
    movsd [rsp+16], xmm2            ; beta2
    movsd [rsp+24], xmm3            ; eps
    
    ; Allocate optimizer struct
    mov rdi, OPT_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed
    mov r15, rax
    
    ; Initialize optimizer
    mov [r15 + OPT_N_PARAMS], r14d
    mov [r15 + OPT_PARAMS], r12
    mov [r15 + OPT_PARAM_NODES], r13
    
    lea rax, [rel adam_step]
    mov [r15 + OPT_STEP_FN], rax
    lea rax, [rel adam_zero_grad]
    mov [r15 + OPT_ZERO_GRAD_FN], rax
    
    ; Allocate Adam state (56 bytes)
    mov rdi, 56
    call mem_alloc
    mov [r15 + OPT_STATE], rax
    mov rbx, rax
    
    ; Copy hyperparameters
    movsd xmm0, [rsp]
    movsd [rbx], xmm0               ; lr
    movsd xmm0, [rsp+8]
    movsd [rbx+8], xmm0             ; beta1
    movsd xmm0, [rsp+16]
    movsd [rbx+16], xmm0            ; beta2
    movsd xmm0, [rsp+24]
    movsd [rbx+24], xmm0            ; eps
    mov qword [rbx+32], 0           ; t = 0
    
    ; Allocate m and v arrays
    mov eax, r14d
    shl eax, 3
    mov edi, eax
    call mem_alloc
    mov [rbx+40], rax               ; m array
    mov [rsp+32], rax
    
    mov eax, r14d
    shl eax, 3
    mov edi, eax
    call mem_alloc
    mov [rbx+48], rax               ; v array
    mov [rsp+40], rax
    
    ; Create m and v tensors (zeros)
    xor ecx, ecx
.create_moments:
    cmp ecx, r14d
    jge .done
    
    push rcx
    mov rax, [r12 + rcx*8]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_zeros
    mov rcx, [rsp]
    mov r8, [rsp+40]                ; m array (adjusted for push)
    mov [r8 + rcx*8], rax
    
    mov rax, [r12 + rcx*8]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_zeros
    pop rcx
    mov r8, [rsp+40]                ; v array
    mov [r8 + rcx*8], rax
    
    inc ecx
    jmp .create_moments

.done:
    mov rax, r15
    
    add rsp, 72
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 72
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; adam_step - Perform one Adam optimization step
; Arguments:
;   RDI = Optimizer* opt
; =============================================================================
adam_step:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 112
    
    mov r12, rdi                    ; optimizer
    
    mov r13d, [r12 + OPT_N_PARAMS]
    mov r14, [r12 + OPT_PARAMS]
    mov r15, [r12 + OPT_PARAM_NODES]
    mov rbx, [r12 + OPT_STATE]
    
    ; Increment time step
    inc qword [rbx+32]
    
    ; Load hyperparameters
    movsd xmm4, [rbx]               ; lr
    movsd xmm5, [rbx+8]             ; beta1
    movsd xmm6, [rbx+16]            ; beta2
    movsd xmm7, [rbx+24]            ; eps
    mov rax, [rbx+32]               ; t
    mov [rsp], rax
    mov rax, [rbx+40]               ; m array
    mov [rsp+8], rax
    mov rax, [rbx+48]               ; v array
    mov [rsp+16], rax
    
    ; Store hyperparams on stack for later use
    movsd [rsp+24], xmm4
    movsd [rsp+32], xmm5
    movsd [rsp+40], xmm6
    movsd [rsp+48], xmm7
    
    ; Compute bias corrections: 1 - beta^t
    ; bias1 = 1 - beta1^t
    movsd xmm0, xmm5                ; beta1
    cvtsi2sd xmm1, qword [rsp]      ; t
    sub rsp, 8
    call pow wrt ..plt
    add rsp, 8
    movsd xmm1, [rel one_f64]
    subsd xmm1, xmm0
    movsd [rsp+56], xmm1            ; 1 - beta1^t
    
    ; bias2 = 1 - beta2^t
    movsd xmm0, [rsp+40]            ; beta2
    cvtsi2sd xmm1, qword [rsp]      ; t
    sub rsp, 8
    call pow wrt ..plt
    add rsp, 8
    movsd xmm1, [rel one_f64]
    subsd xmm1, xmm0
    movsd [rsp+64], xmm1            ; 1 - beta2^t
    
    ; For each parameter
    xor ecx, ecx
.adam_loop:
    cmp ecx, r13d
    jge .done
    
    mov [rsp+72], ecx               ; save index
    
    mov rax, [r14 + rcx*8]          ; param tensor
    mov [rsp+80], rax
    ; Get grad from param_nodes[i]->grad
    mov rsi, [r15 + rcx*8]          ; param_node
    test rsi, rsi
    jz .adam_next                   ; skip if null node
    mov rsi, [rsi + NODE_GRAD]      ; node->grad tensor
    test rsi, rsi
    jz .adam_next                   ; skip if null grad
    mov [rsp+88], rsi
    mov r8, [rsp+8]                 ; m array
    mov rdx, [r8 + rcx*8]           ; m tensor
    mov [rsp+96], rdx
    mov r8, [rsp+16]                ; v array
    mov rax, [r8 + rcx*8]           ; v tensor
    mov [rsp+104], rax
    
    ; Get numel
    mov rdi, [rsp+80]
    call tensor_numel
    mov r10, rax
    
    ; Get data pointers
    mov rax, [rsp+80]
    mov rdi, [rax + TENSOR_DATA]    ; param data
    mov rax, [rsp+88]
    mov r8, [rax + TENSOR_DATA]     ; grad data
    mov rax, [rsp+96]
    mov r9, [rax + TENSOR_DATA]     ; m data
    mov rax, [rsp+104]
    mov r11, [rax + TENSOR_DATA]    ; v data
    
    ; Check dtype
    mov rax, [rsp+80]
    mov eax, [rax + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .adam_f32
    
    ; float64 Adam update
    movsd xmm5, [rsp+32]            ; beta1
    movsd xmm6, [rsp+40]            ; beta2
    
    ; 1 - beta1, 1 - beta2
    movsd xmm3, [rel one_f64]
    movsd xmm2, xmm3
    subsd xmm2, xmm5                ; 1 - beta1
    movsd xmm1, xmm3
    subsd xmm1, xmm6                ; 1 - beta2
    
    xor rcx, rcx
.adam_f64_loop:
    cmp rcx, r10
    jge .adam_next
    
    ; m = beta1 * m + (1 - beta1) * grad
    movsd xmm0, [r9 + rcx*8]        ; m
    mulsd xmm0, xmm5                ; beta1 * m
    movsd xmm4, [r8 + rcx*8]        ; grad
    mulsd xmm4, xmm2                ; (1 - beta1) * grad
    addsd xmm0, xmm4
    movsd [r9 + rcx*8], xmm0        ; store m
    
    ; v = beta2 * v + (1 - beta2) * grad^2
    movsd xmm0, [r11 + rcx*8]       ; v
    mulsd xmm0, xmm6                ; beta2 * v
    movsd xmm4, [r8 + rcx*8]        ; grad
    mulsd xmm4, xmm4                ; grad^2
    mulsd xmm4, xmm1                ; (1 - beta2) * grad^2
    addsd xmm0, xmm4
    movsd [r11 + rcx*8], xmm0       ; store v
    
    ; m_hat = m / (1 - beta1^t)
    movsd xmm0, [r9 + rcx*8]
    divsd xmm0, [rsp+56]
    
    ; v_hat = v / (1 - beta2^t)
    movsd xmm4, [r11 + rcx*8]
    divsd xmm4, [rsp+64]
    
    ; param -= lr * m_hat / (sqrt(v_hat) + eps)
    ; sqrt(v_hat)
    sqrtsd xmm4, xmm4
    addsd xmm4, [rsp+48]            ; + eps
    divsd xmm0, xmm4                ; m_hat / (sqrt(v_hat) + eps)
    mulsd xmm0, [rsp+24]            ; * lr
    
    movsd xmm4, [rdi + rcx*8]       ; param
    subsd xmm4, xmm0
    movsd [rdi + rcx*8], xmm4
    
    inc rcx
    jmp .adam_f64_loop

.adam_f32:
    ; Similar for float32
    movsd xmm5, [rsp+32]
    movsd xmm6, [rsp+40]
    cvtsd2ss xmm5, xmm5
    cvtsd2ss xmm6, xmm6
    
    mov eax, 0x3f800000             ; 1.0f
    movd xmm3, eax
    movaps xmm2, xmm3
    subss xmm2, xmm5
    movaps xmm1, xmm3
    subss xmm1, xmm6
    
    xor rcx, rcx
.adam_f32_loop:
    cmp rcx, r10
    jge .adam_next
    
    movss xmm0, [r9 + rcx*4]
    mulss xmm0, xmm5
    movss xmm4, [r8 + rcx*4]
    mulss xmm4, xmm2
    addss xmm0, xmm4
    movss [r9 + rcx*4], xmm0
    
    movss xmm0, [r11 + rcx*4]
    mulss xmm0, xmm6
    movss xmm4, [r8 + rcx*4]
    mulss xmm4, xmm4
    mulss xmm4, xmm1
    addss xmm0, xmm4
    movss [r11 + rcx*4], xmm0
    
    movss xmm0, [r9 + rcx*4]
    cvtss2sd xmm0, xmm0
    divsd xmm0, [rsp+56]
    cvtsd2ss xmm0, xmm0
    
    movss xmm4, [r11 + rcx*4]
    sqrtss xmm4, xmm4
    movsd xmm7, [rsp+48]
    cvtsd2ss xmm7, xmm7
    addss xmm4, xmm7
    divss xmm0, xmm4
    movsd xmm7, [rsp+24]
    cvtsd2ss xmm7, xmm7
    mulss xmm0, xmm7
    
    movss xmm4, [rdi + rcx*4]
    subss xmm4, xmm0
    movss [rdi + rcx*4], xmm4
    
    inc rcx
    jmp .adam_f32_loop

.adam_next:
    mov ecx, [rsp+72]
    inc ecx
    jmp .adam_loop

.done:
    add rsp, 112
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; adam_zero_grad - Zero all gradients
; Arguments:
;   RDI = Optimizer* opt
; =============================================================================
adam_zero_grad:
    jmp sgd_zero_grad               ; Same implementation

; =============================================================================
; optimizer_free - Free optimizer and its state
; Arguments:
;   RDI = Optimizer* opt
; =============================================================================
optimizer_free:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi
    mov rbx, [r12 + OPT_STATE]
    
    test rbx, rbx
    jz .free_opt
    
    ; Check if SGD (has velocities at offset 16) or Adam (has m at 40)
    ; For simplicity, try to free common patterns
    ; This is a simplified cleanup
    
    mov rdi, rbx
    call mem_free

.free_opt:
    mov rdi, r12
    call mem_free

.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
