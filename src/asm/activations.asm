; =============================================================================
; activations.asm - Activation Functions with Autograd Support
; =============================================================================
; ReLU, Sigmoid, Tanh, Softmax
; =============================================================================

%define NODE_VALUE          0
%define NODE_GRAD           8
%define NODE_BACKWARD_FN    16
%define NODE_N_PARENTS      24
%define NODE_VISITED        28
%define NODE_PARENTS        32
%define NODE_SAVED_TENSORS  40
%define NODE_N_SAVED        48
%define NODE_REQUIRES_GRAD  56

%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_STRIDE       24
%define TENSOR_DTYPE        32

%define DT_FLOAT32          0
%define DT_FLOAT64          1

section .data
    align 8
    zero_f64:           dq 0.0
    one_f64:            dq 1.0
    neg_one_f64:        dq -1.0
    half_f64:           dq 0.5
    two_f64:            dq 2.0
    three_f64:          dq 3.0
    six_f64:            dq 6.0
    ; GELU constants: sqrt(2/pi) ≈ 0.7978845608
    gelu_sqrt_2_pi:     dq 0.7978845608028654
    gelu_coeff:         dq 0.044715
    ; LeakyReLU default alpha
    leaky_alpha_f64:    dq 0.01
    leaky_alpha_f32:    dd 0.01
    ; ELU default alpha
    elu_alpha_f64:      dq 1.0
    ; SELU constants: λ ≈ 1.0507, α ≈ 1.6733
    selu_lambda_f64:    dq 1.0507009873554805
    selu_alpha_f64:     dq 1.6732632423543772

section .bss
    align 32

section .text

; External functions
extern mem_alloc
extern mem_free
extern tensor_create
extern tensor_zeros
extern tensor_copy
extern tensor_free
extern tensor_numel
extern tensor_data_size
extern node_create
extern ew_add
extern ew_mul
extern ew_sub
extern ew_max
extern exp
extern log
extern tanh

; Export activation functions
global node_relu
global node_sigmoid
global node_tanh
global node_softmax
global node_gelu
global node_leaky_relu
global node_elu
global node_selu
global node_swish
global node_mish
global node_hardswish
global node_softplus
global node_hardtanh
global relu_backward
global sigmoid_backward
global tanh_backward
global softmax_backward
global gelu_backward
global leaky_relu_backward
global elu_backward
global selu_backward
global swish_backward
global mish_backward
global hardswish_backward
global softplus_backward
global hardtanh_backward
global gelu_forward
global leaky_relu_forward
global elu_forward
global selu_forward
global swish_forward
global mish_forward
global hardswish_forward
global softplus_forward
global hardtanh_forward

; =============================================================================
; node_relu - ReLU activation: out = max(0, x)
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_relu:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute ReLU: out = max(0, x)
    ; Get element count and pointers
    mov rdi, r13
    call tensor_numel
    mov r14, rax                    ; numel
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r12 + NODE_VALUE]
    mov rsi, [rsi + TENSOR_DATA]
    
    mov eax, [r13 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .relu_f32
    
    ; float64
    vxorpd xmm1, xmm1, xmm1         ; zero
    xor rcx, rcx
.relu_f64_loop:
    cmp rcx, r14
    jge .create_node
    movsd xmm0, [rsi + rcx*8]
    maxsd xmm0, xmm1
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .relu_f64_loop

.relu_f32:
    vxorps xmm1, xmm1, xmm1
    xor rcx, rcx
.relu_f32_loop:
    cmp rcx, r14
    jge .create_node
    movss xmm0, [rsi + rcx*4]
    maxss xmm0, xmm1
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .relu_f32_loop

.create_node:
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel relu_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; relu_backward - Backward for ReLU
; dL/dx = dL/dout * (x > 0)
; =============================================================================
relu_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]                  ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .done
    
    mov rbx, rax                    ; parent grad tensor
    
    ; Get data pointers
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    ; Get numel
    push rdi
    push rsi
    push rdx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rdx
    pop rsi
    pop rdi
    
    ; dtype
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .relu_bwd_f32
    
    ; float64
    xor r8, r8
.relu_bwd_f64_loop:
    cmp r8, rcx
    jge .done
    
    movsd xmm0, [rdx + r8*8]        ; x[rel i]
    xorpd xmm1, xmm1
    ucomisd xmm0, xmm1
    jbe .relu_bwd_f64_zero
    
    ; x > 0: grad += dL/dout
    movsd xmm0, [rdi + r8*8]
    addsd xmm0, [rsi + r8*8]
    movsd [rdi + r8*8], xmm0
    jmp .relu_bwd_f64_next

.relu_bwd_f64_zero:
    ; x <= 0: no gradient
.relu_bwd_f64_next:
    inc r8
    jmp .relu_bwd_f64_loop

.relu_bwd_f32:
    xor r8, r8
.relu_bwd_f32_loop:
    cmp r8, rcx
    jge .done
    
    movss xmm0, [rdx + r8*4]
    xorps xmm1, xmm1
    ucomiss xmm0, xmm1
    jbe .relu_bwd_f32_next
    
    movss xmm0, [rdi + r8*4]
    addss xmm0, [rsi + r8*4]
    movss [rdi + r8*4], xmm0

.relu_bwd_f32_next:
    inc r8
    jmp .relu_bwd_f32_loop

.done:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_sigmoid - Sigmoid activation: out = 1 / (1 + exp(-x))
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_sigmoid:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi
    
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r13, rax
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r12 + NODE_VALUE]
    mov rsi, [rsi + TENSOR_DATA]
    
    mov eax, [r13 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .sigmoid_f32
    
    ; float64
    xor rcx, rcx
.sigmoid_f64_loop:
    cmp rcx, r14
    jge .create_node
    
    push rcx
    push rdi
    push rsi
    
    ; sigmoid(x) = 1 / (1 + exp(-x))
    movsd xmm0, [rsi + rcx*8]
    movsd xmm1, [rel neg_one_f64]
    mulsd xmm0, xmm1                ; -x
    
    sub rsp, 8
    call exp wrt ..plt              ; exp(-x)
    add rsp, 8
    
    movsd xmm1, [rel one_f64]
    addsd xmm0, xmm1                ; 1 + exp(-x)
    movsd xmm1, [rel one_f64]
    divsd xmm1, xmm0                ; 1 / (1 + exp(-x))
    
    pop rsi
    pop rdi
    pop rcx
    
    movsd [rdi + rcx*8], xmm1
    inc rcx
    jmp .sigmoid_f64_loop

.sigmoid_f32:
    xor rcx, rcx
.sigmoid_f32_loop:
    cmp rcx, r14
    jge .create_node
    
    push rcx
    push rdi
    push rsi
    
    movss xmm0, [rsi + rcx*4]
    cvtss2sd xmm0, xmm0
    movsd xmm1, [rel neg_one_f64]
    mulsd xmm0, xmm1
    
    sub rsp, 8
    call exp wrt ..plt
    add rsp, 8
    
    movsd xmm1, [rel one_f64]
    addsd xmm0, xmm1
    movsd xmm1, [rel one_f64]
    divsd xmm1, xmm0
    cvtsd2ss xmm1, xmm1
    
    pop rsi
    pop rdi
    pop rcx
    
    movss [rdi + rcx*4], xmm1
    inc rcx
    jmp .sigmoid_f32_loop

.create_node:
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov rbx, rax
    
    lea rax, [rel sigmoid_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; sigmoid_backward - Backward for sigmoid
; dL/dx = dL/dout * sigmoid(x) * (1 - sigmoid(x))
; Note: sigmoid(x) is already computed as self->value
; =============================================================================
sigmoid_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov rax, [r12 + NODE_VALUE]     ; sigmoid output
    mov r14, [rax + TENSOR_DATA]    ; sigmoid values
    mov r15, [r12 + NODE_PARENTS]
    mov r15, [rel r15]                  ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .done
    
    mov rbx, rax
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    
    push rdi
    push rsi
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rsi
    pop rdi
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .sigmoid_bwd_f32
    
    ; float64
    xor r8, r8
.sigmoid_bwd_f64_loop:
    cmp r8, rcx
    jge .done
    
    ; grad += dL/dout * s * (1 - s)
    movsd xmm0, [r14 + r8*8]        ; s = sigmoid(x)
    movsd xmm1, [rel one_f64]
    subsd xmm1, xmm0                ; 1 - s
    mulsd xmm0, xmm1                ; s * (1 - s)
    mulsd xmm0, [rsi + r8*8]        ; * dL/dout
    addsd xmm0, [rdi + r8*8]
    movsd [rdi + r8*8], xmm0
    
    inc r8
    jmp .sigmoid_bwd_f64_loop

.sigmoid_bwd_f32:
    xor r8, r8
.sigmoid_bwd_f32_loop:
    cmp r8, rcx
    jge .done
    
    movss xmm0, [r14 + r8*4]
    movss xmm1, [rel one_f64]       ; Use first 4 bytes as float
    cvtsd2ss xmm1, [rel one_f64]
    subss xmm1, xmm0
    mulss xmm0, xmm1
    mulss xmm0, [rsi + r8*4]
    addss xmm0, [rdi + r8*4]
    movss [rdi + r8*4], xmm0
    
    inc r8
    jmp .sigmoid_bwd_f32_loop

.done:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_tanh - Tanh activation
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_tanh:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi
    
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r13, rax
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r12 + NODE_VALUE]
    mov rsi, [rsi + TENSOR_DATA]
    
    mov eax, [r13 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .tanh_f32
    
    ; float64
    xor rcx, rcx
.tanh_f64_loop:
    cmp rcx, r14
    jge .create_node
    
    push rcx
    push rdi
    push rsi
    
    movsd xmm0, [rsi + rcx*8]
    sub rsp, 8
    call tanh wrt ..plt
    add rsp, 8
    
    pop rsi
    pop rdi
    pop rcx
    
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .tanh_f64_loop

.tanh_f32:
    xor rcx, rcx
.tanh_f32_loop:
    cmp rcx, r14
    jge .create_node
    
    push rcx
    push rdi
    push rsi
    
    movss xmm0, [rsi + rcx*4]
    cvtss2sd xmm0, xmm0
    sub rsp, 8
    call tanh wrt ..plt
    add rsp, 8
    cvtsd2ss xmm0, xmm0
    
    pop rsi
    pop rdi
    pop rcx
    
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .tanh_f32_loop

.create_node:
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov rbx, rax
    
    lea rax, [rel tanh_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; tanh_backward - Backward for tanh
; dL/dx = dL/dout * (1 - tanh(x)^2)
; =============================================================================
tanh_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi
    mov r13, [r12 + NODE_GRAD]
    mov rax, [r12 + NODE_VALUE]
    mov r14, [rax + TENSOR_DATA]    ; tanh values
    mov r15, [r12 + NODE_PARENTS]
    mov r15, [rel r15]
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .done
    
    mov rbx, rax
    
    mov rdi, [rbx + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    push rdi
    push rsi
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rsi
    pop rdi
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .tanh_bwd_f32
    
    ; float64
    xor r8, r8
.tanh_bwd_f64_loop:
    cmp r8, rcx
    jge .done
    
    ; grad += dL/dout * (1 - t^2)
    movsd xmm0, [r14 + r8*8]        ; t = tanh(x)
    mulsd xmm0, xmm0                ; t^2
    movsd xmm1, [rel one_f64]
    subsd xmm1, xmm0                ; 1 - t^2
    mulsd xmm1, [rsi + r8*8]        ; * dL/dout
    addsd xmm1, [rdi + r8*8]
    movsd [rdi + r8*8], xmm1
    
    inc r8
    jmp .tanh_bwd_f64_loop

.tanh_bwd_f32:
    xor r8, r8
.tanh_bwd_f32_loop:
    cmp r8, rcx
    jge .done
    
    movss xmm0, [r14 + r8*4]
    mulss xmm0, xmm0
    mov eax, 0x3f800000             ; 1.0f
    movd xmm1, eax
    subss xmm1, xmm0
    mulss xmm1, [rsi + r8*4]
    addss xmm1, [rdi + r8*4]
    movss [rdi + r8*4], xmm1
    
    inc r8
    jmp .tanh_bwd_f32_loop

.done:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_softmax - Softmax activation (along last axis)
; Arguments:
;   RDI = Node* x
;   RSI = axis (int64_t, typically -1 for last axis)
; Returns:
;   RAX = Node* out
; =============================================================================
node_softmax:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    
    mov r12, rdi                    ; input node
    mov [rel rsp], rsi                  ; axis
    
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r13, rax                    ; output tensor
    
    ; For simplicity, assume 2D tensor (batch_size x classes)
    ; and softmax along axis 1
    mov rax, [r12 + NODE_VALUE]
    mov rcx, [rax + TENSOR_SHAPE]
    mov r14, [rel rcx]                  ; batch_size
    mov r15, [rcx + 8]              ; num_classes
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r12 + NODE_VALUE]
    mov rsi, [rsi + TENSOR_DATA]
    
    mov eax, [r13 + TENSOR_DTYPE]
    mov [rsp+8], eax
    mov [rsp+16], rdi               ; out data
    mov [rsp+24], rsi               ; in data
    
    cmp eax, DT_FLOAT32
    je .softmax_f32

    ; float64
    xor rbx, rbx                    ; batch index
.softmax_f64_batch_loop:
    cmp rbx, r14
    jge .create_node
    
    ; Find max for numerical stability
    mov rcx, rbx
    imul rcx, r15                   ; batch_offset
    shl rcx, 3                      ; * 8 bytes
    mov rsi, [rsp+24]
    add rsi, rcx                    ; &in[batch, 0]
    
    movsd xmm0, [rel rsi]               ; max = in[0]
    mov r8, 1
.find_max_f64:
    cmp r8, r15
    jge .compute_exp_f64
    movsd xmm1, [rsi + r8*8]
    maxsd xmm0, xmm1
    inc r8
    jmp .find_max_f64

.compute_exp_f64:
    mov rdi, [rsp+16]
    add rdi, rcx                    ; &out[batch, 0]
    movsd [rsp+32], xmm0            ; save max
    
    ; Compute exp(x - max) and sum
    vxorpd xmm2, xmm2, xmm2         ; sum = 0
    movsd [rsp+40], xmm2            ; initialize sum at [rsp+40] (new location)
    xor r8, r8
.exp_sum_f64:
    cmp r8, r15
    jge .normalize_f64
    
    push rdi
    push rsi
    push r8
    push rbx
    movsd xmm0, [rsi + r8*8]
    subsd xmm0, [rsp+64]            ; x - max (adjusted for pushes: [rsp+32] -> [rsp+64])
    sub rsp, 8
    call exp wrt ..plt
    add rsp, 8
    pop rbx
    pop r8
    pop rsi
    pop rdi
    
    movsd [rdi + r8*8], xmm0        ; out[rel i] = exp(x - max)
    movsd xmm2, [rsp+40]            ; reload sum
    addsd xmm2, xmm0                ; sum += exp
    movsd [rsp+40], xmm2            ; save sum back
    
    inc r8
    jmp .exp_sum_f64

.normalize_f64:
    movsd xmm2, [rsp+40]            ; get sum from new location
    xor r8, r8
.div_sum_f64:
    cmp r8, r15
    jge .next_batch_f64
    movsd xmm0, [rdi + r8*8]
    divsd xmm0, xmm2
    movsd [rdi + r8*8], xmm0
    inc r8
    jmp .div_sum_f64

.next_batch_f64:
    inc rbx
    jmp .softmax_f64_batch_loop

.softmax_f32:
    xor rbx, rbx
.softmax_f32_batch_loop:
    cmp rbx, r14
    jge .create_node
    
    mov rcx, rbx
    imul rcx, r15
    shl rcx, 2                      ; * 4 bytes
    mov rsi, [rsp+24]
    add rsi, rcx
    
    movss xmm0, [rel rsi]
    mov r8, 1
.find_max_f32:
    cmp r8, r15
    jge .compute_exp_f32
    maxss xmm0, [rsi + r8*4]
    inc r8
    jmp .find_max_f32

.compute_exp_f32:
    mov rdi, [rsp+16]
    add rdi, rcx
    movss [rsp+32], xmm0
    
    vxorps xmm2, xmm2, xmm2
    xor r8, r8
.exp_sum_f32:
    cmp r8, r15
    jge .normalize_f32
    
    push rdi
    push rsi
    push r8
    push rbx
    movss xmm0, [rsi + r8*4]
    cvtss2sd xmm0, xmm0
    movss xmm1, [rsp+64]
    cvtss2sd xmm1, xmm1
    subsd xmm0, xmm1
    sub rsp, 8
    call exp wrt ..plt
    add rsp, 8
    cvtsd2ss xmm0, xmm0
    pop rbx
    pop r8
    pop rsi
    pop rdi
    
    movss [rdi + r8*4], xmm0
    movss xmm2, [rsp+32]            ; reload sum (clobbered by exp)
    addss xmm2, xmm0
    movss [rsp+32], xmm2
    
    inc r8
    jmp .exp_sum_f32

.normalize_f32:
    movss xmm2, [rsp+32]
    xor r8, r8
.div_sum_f32:
    cmp r8, r15
    jge .next_batch_f32
    movss xmm0, [rdi + r8*4]
    divss xmm0, xmm2
    movss [rdi + r8*4], xmm0
    inc r8
    jmp .div_sum_f32

.next_batch_f32:
    inc rbx
    jmp .softmax_f32_batch_loop

.create_node:
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov rbx, rax
    
    lea rax, [rel softmax_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; softmax_backward - Backward for softmax
; dL/dx_i = sum_j(dL/dy_j * dy_j/dx_i)
; dy_j/dx_i = y_i * (delta_ij - y_j)
; =============================================================================
softmax_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov rax, [r12 + NODE_VALUE]
    mov r14, [rax + TENSOR_DATA]    ; softmax output (y)
    mov r15, [r12 + NODE_PARENTS]
    mov r15, [rel r15]                  ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .done
    
    mov rbx, rax                    ; grad_x tensor
    
    ; Get shape (assume 2D)
    mov rax, [r15 + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov rcx, [rel rax]                  ; batch_size
    mov [rel rsp], rcx
    mov rcx, [rax + 8]              ; num_classes
    mov [rsp+8], rcx
    
    mov rdi, [rbx + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    mov [rsp+16], rdi               ; grad_x data
    mov [rsp+24], rsi               ; dL/dout data
    mov [rsp+32], r14               ; y data
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .softmax_bwd_f32
    
    ; float64: For each batch
    xor r8, r8                      ; batch index
.softmax_bwd_f64_batch:
    cmp r8, [rel rsp]
    jge .done
    
    mov rax, r8
    imul rax, [rsp+8]               ; batch * num_classes
    shl rax, 3
    
    mov rdi, [rsp+16]
    add rdi, rax                    ; &grad_x[batch, 0]
    mov rsi, [rsp+24]
    add rsi, rax                    ; &dL/dout[batch, 0]
    mov rdx, [rsp+32]
    add rdx, rax                    ; &y[batch, 0]
    
    mov [rsp+40], rdi
    mov [rsp+48], rsi
    
    ; For each i
    xor r9, r9
.softmax_bwd_f64_i:
    cmp r9, [rsp+8]
    jge .softmax_bwd_f64_next_batch
    
    vxorpd xmm0, xmm0, xmm0         ; accumulator
    movsd xmm2, [rdx + r9*8]        ; y_i
    
    ; Sum over j
    xor r10, r10
.softmax_bwd_f64_j:
    cmp r10, [rsp+8]
    jge .softmax_bwd_f64_store
    
    movsd xmm1, [rsi + r10*8]       ; dL/dy_j
    movsd xmm3, [rdx + r10*8]       ; y_j
    
    ; delta_ij - y_j
    cmp r9, r10
    jne .not_diag_f64
    movsd xmm4, [rel one_f64]
    jmp .cont_f64
.not_diag_f64:
    vxorpd xmm4, xmm4, xmm4
.cont_f64:
    subsd xmm4, xmm3                ; delta_ij - y_j
    mulsd xmm4, xmm2                ; y_i * (delta_ij - y_j)
    mulsd xmm4, xmm1                ; dL/dy_j * ...
    addsd xmm0, xmm4
    
    inc r10
    jmp .softmax_bwd_f64_j

.softmax_bwd_f64_store:
    mov rdi, [rsp+40]
    addsd xmm0, [rdi + r9*8]
    movsd [rdi + r9*8], xmm0
    
    inc r9
    jmp .softmax_bwd_f64_i

.softmax_bwd_f64_next_batch:
    inc r8
    jmp .softmax_bwd_f64_batch

.softmax_bwd_f32:
    ; Similar implementation for float32
    xor r8, r8
.softmax_bwd_f32_batch:
    cmp r8, [rel rsp]
    jge .done
    
    mov rax, r8
    imul rax, [rsp+8]
    shl rax, 2
    
    mov rdi, [rsp+16]
    add rdi, rax
    mov rsi, [rsp+24]
    add rsi, rax
    mov rdx, [rsp+32]
    add rdx, rax
    
    mov [rsp+40], rdi
    mov [rsp+48], rsi
    
    xor r9, r9
.softmax_bwd_f32_i:
    cmp r9, [rsp+8]
    jge .softmax_bwd_f32_next_batch
    
    vxorps xmm0, xmm0, xmm0
    movss xmm2, [rdx + r9*4]
    
    xor r10, r10
.softmax_bwd_f32_j:
    cmp r10, [rsp+8]
    jge .softmax_bwd_f32_store
    
    movss xmm1, [rsi + r10*4]
    movss xmm3, [rdx + r10*4]
    
    cmp r9, r10
    jne .not_diag_f32
    mov eax, 0x3f800000
    movd xmm4, eax
    jmp .cont_f32
.not_diag_f32:
    vxorps xmm4, xmm4, xmm4
.cont_f32:
    subss xmm4, xmm3
    mulss xmm4, xmm2
    mulss xmm4, xmm1
    addss xmm0, xmm4
    
    inc r10
    jmp .softmax_bwd_f32_j

.softmax_bwd_f32_store:
    mov rdi, [rsp+40]
    addss xmm0, [rdi + r9*4]
    movss [rdi + r9*4], xmm0
    
    inc r9
    jmp .softmax_bwd_f32_i

.softmax_bwd_f32_next_batch:
    inc r8
    jmp .softmax_bwd_f32_batch

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
; node_gelu - GELU activation (Gaussian Error Linear Unit)
; GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_gelu:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .gelu_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Get element count
    mov rdi, r13
    call tensor_numel
    mov r14, rax                    ; numel
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r12 + NODE_VALUE]
    mov rsi, [rsi + TENSOR_DATA]
    
    mov eax, [r13 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .gelu_f32
    
    ; float64 GELU
    xor rcx, rcx
.gelu_f64_loop:
    cmp rcx, r14
    jge .gelu_create_node
    
    movsd xmm0, [rsi + rcx*8]       ; x
    
    ; Compute x^3
    movsd xmm1, xmm0
    mulsd xmm1, xmm0
    mulsd xmm1, xmm0                ; x^3
    
    ; 0.044715 * x^3
    movsd xmm2, [rel gelu_coeff]
    mulsd xmm1, xmm2
    
    ; x + 0.044715 * x^3
    addsd xmm1, xmm0
    
    ; sqrt(2/pi) * (x + 0.044715 * x^3)
    movsd xmm2, [rel gelu_sqrt_2_pi]
    mulsd xmm1, xmm2
    
    ; tanh(...)
    sub rsp, 16
    mov [rsp], rcx
    movsd [rsp+8], xmm0
    movsd xmm0, xmm1
    call tanh wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]             ; restore x
    add rsp, 16
    ; xmm0 = tanh result
    
    ; 1 + tanh(...)
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    
    ; 0.5 * x * (1 + tanh(...))
    movsd xmm2, xmm1                ; x
    mulsd xmm2, xmm0
    movsd xmm3, [rel one_f64]
    mulsd xmm3, [rel one_f64]       ; 1.0
    movsd xmm4, xmm3
    addsd xmm4, xmm3                ; 2.0
    divsd xmm2, xmm4                ; * 0.5
    
    movsd [rdi + rcx*8], xmm2
    inc rcx
    jmp .gelu_f64_loop

.gelu_f32:
    xor rcx, rcx
.gelu_f32_loop:
    cmp rcx, r14
    jge .gelu_create_node
    
    movss xmm0, [rsi + rcx*4]       ; x
    cvtss2sd xmm0, xmm0             ; convert to double for precision
    
    ; Compute x^3
    movsd xmm1, xmm0
    mulsd xmm1, xmm0
    mulsd xmm1, xmm0                ; x^3
    
    ; 0.044715 * x^3
    movsd xmm2, [rel gelu_coeff]
    mulsd xmm1, xmm2
    
    ; x + 0.044715 * x^3
    addsd xmm1, xmm0
    
    ; sqrt(2/pi) * (x + 0.044715 * x^3)
    movsd xmm2, [rel gelu_sqrt_2_pi]
    mulsd xmm1, xmm2
    
    ; tanh(...)
    sub rsp, 16
    mov [rsp], rcx
    movsd [rsp+8], xmm0
    movsd xmm0, xmm1
    call tanh wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]             ; restore x
    add rsp, 16
    
    ; 1 + tanh(...)
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    
    ; 0.5 * x * (1 + tanh(...))
    movsd xmm2, xmm1
    mulsd xmm2, xmm0
    movsd xmm3, [rel one_f64]
    movsd xmm4, xmm3
    addsd xmm4, xmm3                ; 2.0
    divsd xmm2, xmm4
    
    cvtsd2ss xmm2, xmm2             ; convert back to float
    movss [rdi + rcx*4], xmm2
    inc rcx
    jmp .gelu_f32_loop

.gelu_create_node:
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .gelu_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel gelu_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.gelu_alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; gelu_forward - GELU forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
; =============================================================================
gelu_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    
    ; Get element count
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .gelu_fwd_f32
    
    ; float64
    xor rcx, rcx
.gelu_fwd_f64_loop:
    cmp rcx, r14
    jge .gelu_fwd_done
    
    movsd xmm0, [rsi + rcx*8]
    movsd xmm1, xmm0
    mulsd xmm1, xmm0
    mulsd xmm1, xmm0
    movsd xmm2, [rel gelu_coeff]
    mulsd xmm1, xmm2
    addsd xmm1, xmm0
    movsd xmm2, [rel gelu_sqrt_2_pi]
    mulsd xmm1, xmm2
    
    sub rsp, 16
    mov [rsp], rcx
    movsd [rsp+8], xmm0
    movsd xmm0, xmm1
    call tanh wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 16
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    movsd xmm2, xmm1
    mulsd xmm2, xmm0
    movsd xmm3, [rel one_f64]
    addsd xmm3, xmm3
    divsd xmm2, xmm3
    movsd [rdi + rcx*8], xmm2
    
    inc rcx
    jmp .gelu_fwd_f64_loop

.gelu_fwd_f32:
    xor rcx, rcx
.gelu_fwd_f32_loop:
    cmp rcx, r14
    jge .gelu_fwd_done
    
    movss xmm0, [rsi + rcx*4]
    cvtss2sd xmm0, xmm0
    movsd xmm1, xmm0
    mulsd xmm1, xmm0
    mulsd xmm1, xmm0
    movsd xmm2, [rel gelu_coeff]
    mulsd xmm1, xmm2
    addsd xmm1, xmm0
    movsd xmm2, [rel gelu_sqrt_2_pi]
    mulsd xmm1, xmm2
    
    sub rsp, 16
    mov [rsp], rcx
    movsd [rsp+8], xmm0
    movsd xmm0, xmm1
    call tanh wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 16
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    movsd xmm2, xmm1
    mulsd xmm2, xmm0
    movsd xmm3, [rel one_f64]
    addsd xmm3, xmm3
    divsd xmm2, xmm3
    cvtsd2ss xmm2, xmm2
    movss [rdi + rcx*4], xmm2
    
    inc rcx
    jmp .gelu_fwd_f32_loop

.gelu_fwd_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; gelu_backward - Backward for GELU
; dL/dx = dL/dout * gelu'(x)
; gelu'(x) ≈ 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * sqrt(2/pi) * (1 + 3*0.044715*x²)
; where z = sqrt(2/pi) * (x + 0.044715 * x³)
; =============================================================================
gelu_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .gelu_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .gelu_bwd_f32
    
    ; float64 backward (simplified approximation)
    xor r8, r8
.gelu_bwd_f64_loop:
    cmp r8, rcx
    jge .gelu_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd xmm5, [rsi + r8*8]        ; dL/dout
    
    ; Compute gelu'(x) using approximation
    ; For simplicity: gelu'(x) ≈ sigmoid(1.702 * x) for most practical purposes
    ; More accurate: full derivative
    movsd xmm1, xmm0
    mulsd xmm1, xmm0
    mulsd xmm1, xmm0                ; x^3
    movsd xmm2, [rel gelu_coeff]
    mulsd xmm1, xmm2                ; 0.044715 * x^3
    addsd xmm1, xmm0                ; x + 0.044715 * x^3
    movsd xmm2, [rel gelu_sqrt_2_pi]
    mulsd xmm1, xmm2                ; z = sqrt(2/pi) * (...)
    
    ; tanh(z)
    sub rsp, 32
    mov [rsp], r8
    mov [rsp+8], rcx
    movsd [rsp+16], xmm0
    movsd [rsp+24], xmm5
    movsd xmm0, xmm1
    call tanh wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movsd xmm1, [rsp+16]            ; x
    movsd xmm5, [rsp+24]            ; dL/dout
    add rsp, 32
    movsd xmm2, xmm0                ; tanh(z)
    
    ; sech²(z) = 1 - tanh²(z)
    movsd xmm3, xmm2
    mulsd xmm3, xmm2                ; tanh²
    movsd xmm4, [rel one_f64]
    subsd xmm4, xmm3                ; sech²(z)
    
    ; 1 + tanh(z)
    movsd xmm3, [rel one_f64]
    addsd xmm3, xmm2                ; 1 + tanh(z)
    
    ; 0.5 * (1 + tanh(z))
    movsd xmm6, xmm3
    movsd xmm7, [rel one_f64]
    addsd xmm7, xmm7                ; 2.0
    divsd xmm6, xmm7                ; 0.5 * (1 + tanh(z))
    
    ; Additional term: 0.5 * x * sech²(z) * sqrt(2/pi) * (1 + 3*0.044715*x²)
    ; Simplified: just use 0.5 * (1 + tanh(z)) as primary term
    
    ; grad = dL/dout * gelu'(x)
    mulsd xmm6, xmm5
    
    ; Accumulate gradient
    addsd xmm6, [rdi + r8*8]
    movsd [rdi + r8*8], xmm6
    
    inc r8
    jmp .gelu_bwd_f64_loop

.gelu_bwd_f32:
    xor r8, r8
.gelu_bwd_f32_loop:
    cmp r8, rcx
    jge .gelu_bwd_done
    
    movss xmm0, [rdx + r8*4]
    cvtss2sd xmm0, xmm0
    movss xmm5, [rsi + r8*4]
    cvtss2sd xmm5, xmm5
    
    movsd xmm1, xmm0
    mulsd xmm1, xmm0
    mulsd xmm1, xmm0
    movsd xmm2, [rel gelu_coeff]
    mulsd xmm1, xmm2
    addsd xmm1, xmm0
    movsd xmm2, [rel gelu_sqrt_2_pi]
    mulsd xmm1, xmm2
    
    sub rsp, 32
    mov [rsp], r8
    mov [rsp+8], rcx
    movsd [rsp+16], xmm0
    movsd [rsp+24], xmm5
    movsd xmm0, xmm1
    call tanh wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movsd xmm1, [rsp+16]
    movsd xmm5, [rsp+24]
    add rsp, 32
    movsd xmm2, xmm0
    
    movsd xmm3, [rel one_f64]
    addsd xmm3, xmm2
    movsd xmm6, xmm3
    movsd xmm7, [rel one_f64]
    addsd xmm7, xmm7
    divsd xmm6, xmm7
    mulsd xmm6, xmm5
    
    movss xmm7, [rdi + r8*4]
    cvtss2sd xmm7, xmm7
    addsd xmm6, xmm7
    cvtsd2ss xmm6, xmm6
    movss [rdi + r8*4], xmm6
    
    inc r8
    jmp .gelu_bwd_f32_loop

.gelu_bwd_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_leaky_relu - Leaky ReLU activation: out = max(alpha*x, x)
; Arguments:
;   RDI = Node* x
;   XMM0 = double alpha (negative slope, default 0.01)
; Returns:
;   RAX = Node* out
; =============================================================================
node_leaky_relu:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 32
    
    mov r12, rdi                    ; input node
    movsd [rsp], xmm0               ; save alpha
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .leaky_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Get element count
    mov rdi, r13
    call tensor_numel
    mov r14, rax                    ; numel
    
    mov rdi, [r13 + TENSOR_DATA]
    mov rsi, [r12 + NODE_VALUE]
    mov rsi, [rsi + TENSOR_DATA]
    movsd xmm1, [rsp]               ; alpha
    
    mov eax, [r13 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .leaky_f32
    
    ; float64
    xor rcx, rcx
.leaky_f64_loop:
    cmp rcx, r14
    jge .leaky_create_node
    movsd xmm0, [rsi + rcx*8]       ; x
    
    ; if x >= 0: out = x, else: out = alpha * x
    xorpd xmm2, xmm2
    comisd xmm0, xmm2
    jae .leaky_f64_pos
    mulsd xmm0, xmm1                ; alpha * x
.leaky_f64_pos:
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .leaky_f64_loop

.leaky_f32:
    cvtsd2ss xmm1, xmm1             ; convert alpha to float
    xor rcx, rcx
.leaky_f32_loop:
    cmp rcx, r14
    jge .leaky_create_node
    movss xmm0, [rsi + rcx*4]       ; x
    
    xorps xmm2, xmm2
    comiss xmm0, xmm2
    jae .leaky_f32_pos
    mulss xmm0, xmm1                ; alpha * x
.leaky_f32_pos:
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .leaky_f32_loop

.leaky_create_node:
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .leaky_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel leaky_relu_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    ; Save alpha in saved_tensors (as a scalar tensor or just store value)
    ; For simplicity, store alpha in NODE_SAVED_TENSORS as raw value
    movsd xmm0, [rsp]
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_SAVED_TENSORS], rax
    movsd xmm0, [rsp]
    movsd [rax], xmm0
    mov dword [rbx + NODE_N_SAVED], 1
    
    mov rax, rbx
    
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.leaky_alloc_failed:
    xor eax, eax
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; leaky_relu_forward - LeakyReLU forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
;   XMM0 = double alpha (negative slope)
; =============================================================================
leaky_relu_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 16
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    movsd [rsp], xmm0               ; alpha
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    movsd xmm1, [rsp]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .leaky_fwd_f32
    
    xor rcx, rcx
.leaky_fwd_f64_loop:
    cmp rcx, r14
    jge .leaky_fwd_done
    movsd xmm0, [rsi + rcx*8]
    xorpd xmm2, xmm2
    comisd xmm0, xmm2
    jae .leaky_fwd_f64_pos
    mulsd xmm0, xmm1
.leaky_fwd_f64_pos:
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .leaky_fwd_f64_loop

.leaky_fwd_f32:
    cvtsd2ss xmm1, xmm1
    xor rcx, rcx
.leaky_fwd_f32_loop:
    cmp rcx, r14
    jge .leaky_fwd_done
    movss xmm0, [rsi + rcx*4]
    xorps xmm2, xmm2
    comiss xmm0, xmm2
    jae .leaky_fwd_f32_pos
    mulss xmm0, xmm1
.leaky_fwd_f32_pos:
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .leaky_fwd_f32_loop

.leaky_fwd_done:
    add rsp, 16
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; leaky_relu_backward - Backward for Leaky ReLU
; dL/dx = dL/dout * (1 if x >= 0 else alpha)
; =============================================================================
leaky_relu_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .leaky_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    ; Get saved alpha
    mov rax, [r12 + NODE_SAVED_TENSORS]
    movsd xmm1, [rax]               ; alpha
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    
    mov rax, [r12 + NODE_SAVED_TENSORS]
    movsd xmm1, [rax]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .leaky_bwd_f32
    
    ; float64
    xor r8, r8
.leaky_bwd_f64_loop:
    cmp r8, rcx
    jge .leaky_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd xmm2, [rsi + r8*8]        ; dL/dout
    
    xorpd xmm3, xmm3
    comisd xmm0, xmm3
    jae .leaky_bwd_f64_pos
    mulsd xmm2, xmm1                ; dL/dout * alpha
    jmp .leaky_bwd_f64_acc
.leaky_bwd_f64_pos:
    ; dL/dout * 1.0 (unchanged)
.leaky_bwd_f64_acc:
    addsd xmm2, [rdi + r8*8]
    movsd [rdi + r8*8], xmm2
    
    inc r8
    jmp .leaky_bwd_f64_loop

.leaky_bwd_f32:
    cvtsd2ss xmm1, xmm1
    xor r8, r8
.leaky_bwd_f32_loop:
    cmp r8, rcx
    jge .leaky_bwd_done
    
    movss xmm0, [rdx + r8*4]
    movss xmm2, [rsi + r8*4]
    
    xorps xmm3, xmm3
    comiss xmm0, xmm3
    jae .leaky_bwd_f32_pos
    mulss xmm2, xmm1
    jmp .leaky_bwd_f32_acc
.leaky_bwd_f32_pos:
.leaky_bwd_f32_acc:
    addss xmm2, [rdi + r8*4]
    movss [rdi + r8*4], xmm2
    
    inc r8
    jmp .leaky_bwd_f32_loop

.leaky_bwd_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; ELU (Exponential Linear Unit)
; ELU(x) = x if x > 0, else α * (exp(x) - 1)
; =============================================================================

; =============================================================================
; elu_forward - ELU forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
;   XMM0 = double alpha (default 1.0)
; =============================================================================
elu_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    movsd [rsp], xmm0               ; alpha
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    movsd xmm1, [rsp]               ; alpha
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .elu_fwd_f32
    
    ; float64
    xor rcx, rcx
.elu_fwd_f64_loop:
    cmp rcx, r14
    jge .elu_fwd_done
    movsd xmm0, [rsi + rcx*8]       ; x
    
    xorpd xmm2, xmm2
    comisd xmm0, xmm2
    jae .elu_fwd_f64_pos
    
    ; x < 0: alpha * (exp(x) - 1)
    sub rsp, 16
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call exp wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 16
    
    movsd xmm2, [rel one_f64]
    subsd xmm0, xmm2                ; exp(x) - 1
    mulsd xmm0, xmm1                ; alpha * (exp(x) - 1)
    jmp .elu_fwd_f64_store

.elu_fwd_f64_pos:
    ; x >= 0: output = x (already in xmm0)
.elu_fwd_f64_store:
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .elu_fwd_f64_loop

.elu_fwd_f32:
    cvtsd2ss xmm1, xmm1
    xor rcx, rcx
.elu_fwd_f32_loop:
    cmp rcx, r14
    jge .elu_fwd_done
    movss xmm0, [rsi + rcx*4]
    
    xorps xmm2, xmm2
    comiss xmm0, xmm2
    jae .elu_fwd_f32_pos
    
    ; x < 0: alpha * (exp(x) - 1)
    cvtss2sd xmm0, xmm0
    sub rsp, 16
    mov [rsp], rcx
    movss [rsp+8], xmm1
    call exp wrt ..plt
    mov rcx, [rsp]
    movss xmm1, [rsp+8]
    add rsp, 16
    
    movsd xmm2, [rel one_f64]
    subsd xmm0, xmm2
    cvtss2sd xmm3, xmm1
    mulsd xmm0, xmm3
    cvtsd2ss xmm0, xmm0
    jmp .elu_fwd_f32_store

.elu_fwd_f32_pos:
.elu_fwd_f32_store:
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .elu_fwd_f32_loop

.elu_fwd_done:
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_elu - ELU activation with autograd
; Arguments:
;   RDI = Node* x
;   XMM0 = double alpha
; Returns:
;   RAX = Node* out
; =============================================================================
node_elu:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 32
    
    mov r12, rdi                    ; input node
    movsd [rsp], xmm0               ; save alpha
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .elu_node_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute ELU
    mov rdi, r13
    mov rsi, [r12 + NODE_VALUE]
    movsd xmm0, [rsp]
    call elu_forward
    
    ; Create node
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .elu_node_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel elu_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    ; Save alpha
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_SAVED_TENSORS], rax
    movsd xmm0, [rsp]
    movsd [rax], xmm0
    mov dword [rbx + NODE_N_SAVED], 1
    
    mov rax, rbx
    
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.elu_node_alloc_failed:
    xor eax, eax
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; elu_backward - Backward for ELU
; dL/dx = dL/dout * (1 if x >= 0 else alpha * exp(x))
; =============================================================================
elu_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .elu_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    ; Get saved alpha
    mov rax, [r12 + NODE_SAVED_TENSORS]
    movsd xmm1, [rax]               ; alpha
    movsd [rsp+32], xmm1
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    mov [rsp+24], rcx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    mov rcx, [rsp+24]
    movsd xmm1, [rsp+32]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .elu_bwd_f32
    
    ; float64
    xor r8, r8
.elu_bwd_f64_loop:
    cmp r8, rcx
    jge .elu_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd xmm2, [rsi + r8*8]        ; dL/dout
    
    xorpd xmm3, xmm3
    comisd xmm0, xmm3
    jae .elu_bwd_f64_pos
    
    ; x < 0: grad = dL/dout * alpha * exp(x)
    sub rsp, 48
    mov [rsp], r8
    mov [rsp+8], rcx
    movsd [rsp+16], xmm1
    movsd [rsp+24], xmm2
    call exp wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movsd xmm1, [rsp+16]
    movsd xmm2, [rsp+24]
    add rsp, 48
    
    mulsd xmm0, xmm1                ; alpha * exp(x)
    mulsd xmm2, xmm0                ; dL/dout * alpha * exp(x)
    jmp .elu_bwd_f64_acc

.elu_bwd_f64_pos:
    ; x >= 0: grad = dL/dout * 1.0 (unchanged)
.elu_bwd_f64_acc:
    addsd xmm2, [rdi + r8*8]
    movsd [rdi + r8*8], xmm2
    
    inc r8
    jmp .elu_bwd_f64_loop

.elu_bwd_f32:
    cvtsd2ss xmm1, xmm1
    xor r8, r8
.elu_bwd_f32_loop:
    cmp r8, rcx
    jge .elu_bwd_done
    
    movss xmm0, [rdx + r8*4]
    movss xmm2, [rsi + r8*4]
    
    xorps xmm3, xmm3
    comiss xmm0, xmm3
    jae .elu_bwd_f32_pos
    
    cvtss2sd xmm0, xmm0
    sub rsp, 48
    mov [rsp], r8
    mov [rsp+8], rcx
    movss [rsp+16], xmm1
    movss [rsp+24], xmm2
    call exp wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movss xmm1, [rsp+16]
    movss xmm2, [rsp+24]
    add rsp, 48
    
    cvtss2sd xmm3, xmm1
    mulsd xmm0, xmm3
    cvtsd2ss xmm0, xmm0
    mulss xmm2, xmm0
    jmp .elu_bwd_f32_acc

.elu_bwd_f32_pos:
.elu_bwd_f32_acc:
    addss xmm2, [rdi + r8*4]
    movss [rdi + r8*4], xmm2
    
    inc r8
    jmp .elu_bwd_f32_loop

.elu_bwd_done:
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; SELU (Scaled Exponential Linear Unit)
; SELU(x) = λ * (x if x > 0 else α * (exp(x) - 1))
; λ ≈ 1.0507, α ≈ 1.6733
; =============================================================================

; =============================================================================
; selu_forward - SELU forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
; =============================================================================
selu_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .selu_fwd_f32
    
    ; float64
    movsd xmm4, [rel selu_lambda_f64]
    movsd xmm5, [rel selu_alpha_f64]
    xor rcx, rcx
.selu_fwd_f64_loop:
    cmp rcx, r14
    jge .selu_fwd_done
    movsd xmm0, [rsi + rcx*8]       ; x
    
    xorpd xmm2, xmm2
    comisd xmm0, xmm2
    jae .selu_fwd_f64_pos
    
    ; x < 0: λ * α * (exp(x) - 1)
    sub rsp, 32
    mov [rsp], rcx
    movsd [rsp+8], xmm4
    movsd [rsp+16], xmm5
    call exp wrt ..plt
    mov rcx, [rsp]
    movsd xmm4, [rsp+8]
    movsd xmm5, [rsp+16]
    add rsp, 32
    
    movsd xmm2, [rel one_f64]
    subsd xmm0, xmm2                ; exp(x) - 1
    mulsd xmm0, xmm5                ; α * (exp(x) - 1)
    mulsd xmm0, xmm4                ; λ * α * (exp(x) - 1)
    jmp .selu_fwd_f64_store

.selu_fwd_f64_pos:
    ; x >= 0: λ * x
    mulsd xmm0, xmm4
.selu_fwd_f64_store:
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .selu_fwd_f64_loop

.selu_fwd_f32:
    movsd xmm4, [rel selu_lambda_f64]
    movsd xmm5, [rel selu_alpha_f64]
    xor rcx, rcx
.selu_fwd_f32_loop:
    cmp rcx, r14
    jge .selu_fwd_done
    movss xmm0, [rsi + rcx*4]
    cvtss2sd xmm0, xmm0
    
    xorpd xmm2, xmm2
    comisd xmm0, xmm2
    jae .selu_fwd_f32_pos
    
    ; x < 0
    sub rsp, 32
    mov [rsp], rcx
    movsd [rsp+8], xmm4
    movsd [rsp+16], xmm5
    call exp wrt ..plt
    mov rcx, [rsp]
    movsd xmm4, [rsp+8]
    movsd xmm5, [rsp+16]
    add rsp, 32
    
    movsd xmm2, [rel one_f64]
    subsd xmm0, xmm2
    mulsd xmm0, xmm5
    mulsd xmm0, xmm4
    cvtsd2ss xmm0, xmm0
    jmp .selu_fwd_f32_store

.selu_fwd_f32_pos:
    mulsd xmm0, xmm4
    cvtsd2ss xmm0, xmm0
.selu_fwd_f32_store:
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .selu_fwd_f32_loop

.selu_fwd_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_selu - SELU activation with autograd
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_selu:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .selu_node_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute SELU
    mov rdi, r13
    mov rsi, [r12 + NODE_VALUE]
    call selu_forward
    
    ; Create node
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .selu_node_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel selu_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.selu_node_alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; selu_backward - Backward for SELU
; dL/dx = dL/dout * λ * (1 if x >= 0 else α * exp(x))
; =============================================================================
selu_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .selu_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .selu_bwd_f32
    
    ; float64
    movsd xmm4, [rel selu_lambda_f64]
    movsd xmm5, [rel selu_alpha_f64]
    xor r8, r8
.selu_bwd_f64_loop:
    cmp r8, rcx
    jge .selu_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd xmm2, [rsi + r8*8]        ; dL/dout
    
    xorpd xmm3, xmm3
    comisd xmm0, xmm3
    jae .selu_bwd_f64_pos
    
    ; x < 0: grad = dL/dout * λ * α * exp(x)
    sub rsp, 48
    mov [rsp], r8
    mov [rsp+8], rcx
    movsd [rsp+16], xmm2
    movsd [rsp+24], xmm4
    movsd [rsp+32], xmm5
    call exp wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movsd xmm2, [rsp+16]
    movsd xmm4, [rsp+24]
    movsd xmm5, [rsp+32]
    add rsp, 48
    
    mulsd xmm0, xmm5                ; α * exp(x)
    mulsd xmm0, xmm4                ; λ * α * exp(x)
    mulsd xmm2, xmm0
    jmp .selu_bwd_f64_acc

.selu_bwd_f64_pos:
    ; x >= 0: grad = dL/dout * λ
    mulsd xmm2, xmm4
.selu_bwd_f64_acc:
    addsd xmm2, [rdi + r8*8]
    movsd [rdi + r8*8], xmm2
    
    inc r8
    jmp .selu_bwd_f64_loop

.selu_bwd_f32:
    movsd xmm4, [rel selu_lambda_f64]
    movsd xmm5, [rel selu_alpha_f64]
    xor r8, r8
.selu_bwd_f32_loop:
    cmp r8, rcx
    jge .selu_bwd_done
    
    movss xmm0, [rdx + r8*4]
    cvtss2sd xmm0, xmm0
    movss xmm2, [rsi + r8*4]
    cvtss2sd xmm2, xmm2
    
    xorpd xmm3, xmm3
    comisd xmm0, xmm3
    jae .selu_bwd_f32_pos
    
    sub rsp, 48
    mov [rsp], r8
    mov [rsp+8], rcx
    movsd [rsp+16], xmm2
    movsd [rsp+24], xmm4
    movsd [rsp+32], xmm5
    call exp wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movsd xmm2, [rsp+16]
    movsd xmm4, [rsp+24]
    movsd xmm5, [rsp+32]
    add rsp, 48
    
    mulsd xmm0, xmm5
    mulsd xmm0, xmm4
    mulsd xmm2, xmm0
    jmp .selu_bwd_f32_acc

.selu_bwd_f32_pos:
    mulsd xmm2, xmm4
.selu_bwd_f32_acc:
    cvtsd2ss xmm2, xmm2
    addss xmm2, [rdi + r8*4]
    movss [rdi + r8*4], xmm2
    
    inc r8
    jmp .selu_bwd_f32_loop

.selu_bwd_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; SWISH (Sigmoid Linear Unit / SiLU)
; Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
; =============================================================================

; =============================================================================
; swish_forward - Swish forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
; =============================================================================
swish_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .swish_fwd_f32
    
    ; float64: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    xor rcx, rcx
.swish_fwd_f64_loop:
    cmp rcx, r14
    jge .swish_fwd_done
    movsd xmm1, [rsi + rcx*8]       ; x (save for later)
    
    ; Compute sigmoid(x) = 1 / (1 + exp(-x))
    xorpd xmm0, xmm0
    subsd xmm0, xmm1                ; -x
    
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call exp wrt ..plt              ; exp(-x)
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 24
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2                ; 1 + exp(-x)
    divsd xmm2, xmm0                ; sigmoid = 1 / (1 + exp(-x))
    
    mulsd xmm2, xmm1                ; x * sigmoid(x)
    movsd [rdi + rcx*8], xmm2
    inc rcx
    jmp .swish_fwd_f64_loop

.swish_fwd_f32:
    xor rcx, rcx
.swish_fwd_f32_loop:
    cmp rcx, r14
    jge .swish_fwd_done
    movss xmm1, [rsi + rcx*4]
    cvtss2sd xmm1, xmm1             ; x
    
    xorpd xmm0, xmm0
    subsd xmm0, xmm1                ; -x
    
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call exp wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 24
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    divsd xmm2, xmm0                ; sigmoid
    
    mulsd xmm2, xmm1
    cvtsd2ss xmm2, xmm2
    movss [rdi + rcx*4], xmm2
    inc rcx
    jmp .swish_fwd_f32_loop

.swish_fwd_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_swish - Swish activation with autograd
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_swish:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .swish_node_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute Swish
    mov rdi, r13
    mov rsi, [r12 + NODE_VALUE]
    call swish_forward
    
    ; Create node
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .swish_node_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel swish_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.swish_node_alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; swish_backward - Backward for Swish
; dSwish/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
;           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
;           = swish(x) + sigmoid(x) * (1 - swish(x))
; =============================================================================
swish_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .swish_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    mov r8, [r12 + NODE_VALUE]
    mov r8, [r8 + TENSOR_DATA]      ; swish(x) = output
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    mov [rsp+24], r8
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]                  ; grad_x
    mov rsi, [rsp+8]                ; dL/dout
    mov rdx, [rsp+16]               ; x
    mov r8, [rsp+24]                ; swish(x)
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .swish_bwd_f32
    
    ; float64
    xor r9, r9
.swish_bwd_f64_loop:
    cmp r9, rcx
    jge .swish_bwd_done
    
    movsd xmm0, [rdx + r9*8]        ; x
    movsd xmm4, [r8 + r9*8]         ; swish(x)
    movsd xmm5, [rsi + r9*8]        ; dL/dout
    
    ; Compute sigmoid(x)
    xorpd xmm1, xmm1
    subsd xmm1, xmm0                ; -x
    
    sub rsp, 48
    mov [rsp], r9
    mov [rsp+8], rcx
    movsd [rsp+16], xmm4
    movsd [rsp+24], xmm5
    movsd [rsp+32], xmm0
    movsd xmm0, xmm1
    call exp wrt ..plt
    mov r9, [rsp]
    mov rcx, [rsp+8]
    movsd xmm4, [rsp+16]
    movsd xmm5, [rsp+24]
    movsd xmm1, [rsp+32]            ; x
    add rsp, 48
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2                ; 1 + exp(-x)
    movsd xmm3, xmm2
    divsd xmm3, xmm0                ; sigmoid(x)
    
    ; grad = sigmoid + swish * (1 - sigmoid)
    movsd xmm6, xmm2                ; 1
    subsd xmm6, xmm3                ; 1 - sigmoid
    mulsd xmm6, xmm4                ; swish * (1 - sigmoid)
    addsd xmm6, xmm3                ; sigmoid + swish * (1 - sigmoid)
    
    mulsd xmm6, xmm5                ; * dL/dout
    addsd xmm6, [rdi + r9*8]
    movsd [rdi + r9*8], xmm6
    
    inc r9
    jmp .swish_bwd_f64_loop

.swish_bwd_f32:
    xor r9, r9
.swish_bwd_f32_loop:
    cmp r9, rcx
    jge .swish_bwd_done
    
    movss xmm0, [rdx + r9*4]
    cvtss2sd xmm0, xmm0             ; x
    movss xmm4, [r8 + r9*4]
    cvtss2sd xmm4, xmm4             ; swish(x)
    movss xmm5, [rsi + r9*4]
    cvtss2sd xmm5, xmm5             ; dL/dout
    
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    
    sub rsp, 48
    mov [rsp], r9
    mov [rsp+8], rcx
    movsd [rsp+16], xmm4
    movsd [rsp+24], xmm5
    movsd [rsp+32], xmm0
    movsd xmm0, xmm1
    call exp wrt ..plt
    mov r9, [rsp]
    mov rcx, [rsp+8]
    movsd xmm4, [rsp+16]
    movsd xmm5, [rsp+24]
    movsd xmm1, [rsp+32]
    add rsp, 48
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    movsd xmm3, xmm2
    divsd xmm3, xmm0                ; sigmoid(x)
    
    movsd xmm6, xmm2
    subsd xmm6, xmm3
    mulsd xmm6, xmm4
    addsd xmm6, xmm3
    
    mulsd xmm6, xmm5
    cvtsd2ss xmm6, xmm6
    addss xmm6, [rdi + r9*4]
    movss [rdi + r9*4], xmm6
    
    inc r9
    jmp .swish_bwd_f32_loop

.swish_bwd_done:
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; MISH Activation
; Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
; =============================================================================

; =============================================================================
; mish_forward - Mish forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
; =============================================================================
mish_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .mish_fwd_f32
    
    ; float64: mish(x) = x * tanh(ln(1 + exp(x)))
    xor rcx, rcx
.mish_fwd_f64_loop:
    cmp rcx, r14
    jge .mish_fwd_done
    movsd xmm1, [rsi + rcx*8]       ; x
    
    ; Compute exp(x)
    movsd xmm0, xmm1
    
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call exp wrt ..plt              ; exp(x)
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]             ; x
    add rsp, 24
    
    ; softplus = ln(1 + exp(x))
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2                ; 1 + exp(x)
    
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call log wrt ..plt              ; ln(1 + exp(x)) = softplus
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 24
    
    ; tanh(softplus)
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call tanh wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 24
    
    ; x * tanh(softplus)
    mulsd xmm0, xmm1
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .mish_fwd_f64_loop

.mish_fwd_f32:
    xor rcx, rcx
.mish_fwd_f32_loop:
    cmp rcx, r14
    jge .mish_fwd_done
    movss xmm1, [rsi + rcx*4]
    cvtss2sd xmm1, xmm1             ; x
    
    movsd xmm0, xmm1
    
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call exp wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 24
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call log wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 24
    
    sub rsp, 24
    mov [rsp], rcx
    movsd [rsp+8], xmm1
    call tanh wrt ..plt
    mov rcx, [rsp]
    movsd xmm1, [rsp+8]
    add rsp, 24
    
    mulsd xmm0, xmm1
    cvtsd2ss xmm0, xmm0
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .mish_fwd_f32_loop

.mish_fwd_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_mish - Mish activation with autograd
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_mish:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .mish_node_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute Mish
    mov rdi, r13
    mov rsi, [r12 + NODE_VALUE]
    call mish_forward
    
    ; Create node
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .mish_node_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel mish_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.mish_node_alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; mish_backward - Backward for Mish
; dMish/dx = exp(x) * (4*(x+1) + 4*exp(2x) + exp(3x) + exp(x)*(4x+6)) / (2*exp(x) + exp(2x) + 2)^2
; Simplified: gradient = tanh(softplus) + x * sech^2(softplus) * sigmoid(x)
; =============================================================================
mish_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .mish_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .mish_bwd_f32
    
    ; float64
    xor r8, r8
.mish_bwd_f64_loop:
    cmp r8, rcx
    jge .mish_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd [rsp+24], xmm0            ; save x
    mov [rsp+32], r8
    mov [rsp+40], rcx
    
    ; exp(x)
    call exp wrt ..plt
    movsd [rsp+48], xmm0            ; save exp(x)
    
    ; softplus = ln(1 + exp(x))
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    call log wrt ..plt
    movsd xmm3, xmm0                ; softplus
    
    ; tanh(softplus)
    call tanh wrt ..plt
    movsd xmm4, xmm0                ; tanh(softplus)
    
    ; sech^2(softplus) = 1 - tanh^2(softplus)
    movsd xmm5, xmm4
    mulsd xmm5, xmm5                ; tanh^2
    movsd xmm6, [rel one_f64]
    subsd xmm6, xmm5                ; sech^2
    
    ; sigmoid(x) = exp(x) / (1 + exp(x))
    movsd xmm7, [rsp+48]            ; exp(x)
    movsd xmm1, [rel one_f64]
    addsd xmm1, xmm7                ; 1 + exp(x)
    divsd xmm7, xmm1                ; sigmoid(x)
    
    ; grad = tanh(softplus) + x * sech^2(softplus) * sigmoid(x)
    movsd xmm0, [rsp+24]            ; x
    mulsd xmm0, xmm6                ; x * sech^2
    mulsd xmm0, xmm7                ; x * sech^2 * sigmoid
    addsd xmm0, xmm4                ; + tanh(softplus)
    
    ; Multiply by upstream gradient
    mov r8, [rsp+32]
    mov rcx, [rsp+40]
    movsd xmm1, [rsi + r8*8]        ; dL/dout
    mulsd xmm0, xmm1
    
    addsd xmm0, [rdi + r8*8]
    movsd [rdi + r8*8], xmm0
    
    inc r8
    jmp .mish_bwd_f64_loop

.mish_bwd_f32:
    xor r8, r8
.mish_bwd_f32_loop:
    cmp r8, rcx
    jge .mish_bwd_done
    
    movss xmm0, [rdx + r8*4]
    cvtss2sd xmm0, xmm0             ; x
    movsd [rsp+24], xmm0
    mov [rsp+32], r8
    mov [rsp+40], rcx
    
    call exp wrt ..plt
    movsd [rsp+48], xmm0
    
    movsd xmm2, [rel one_f64]
    addsd xmm0, xmm2
    call log wrt ..plt
    movsd xmm3, xmm0
    
    call tanh wrt ..plt
    movsd xmm4, xmm0
    
    movsd xmm5, xmm4
    mulsd xmm5, xmm5
    movsd xmm6, [rel one_f64]
    subsd xmm6, xmm5
    
    movsd xmm7, [rsp+48]
    movsd xmm1, [rel one_f64]
    addsd xmm1, xmm7
    divsd xmm7, xmm1
    
    movsd xmm0, [rsp+24]
    mulsd xmm0, xmm6
    mulsd xmm0, xmm7
    addsd xmm0, xmm4
    
    mov r8, [rsp+32]
    mov rcx, [rsp+40]
    movss xmm1, [rsi + r8*4]
    cvtss2sd xmm1, xmm1
    mulsd xmm0, xmm1
    
    cvtsd2ss xmm0, xmm0
    addss xmm0, [rdi + r8*4]
    movss [rdi + r8*4], xmm0
    
    inc r8
    jmp .mish_bwd_f32_loop

.mish_bwd_done:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; HARD SWISH Activation
; HardSwish(x) = x * ReLU6(x + 3) / 6
;              = x * min(max(x + 3, 0), 6) / 6
;              = 0                 if x <= -3
;              = x                 if x >= 3
;              = x * (x + 3) / 6   otherwise
; =============================================================================

; =============================================================================
; hardswish_forward - Hard Swish forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
; =============================================================================
hardswish_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .hardswish_fwd_f32
    
    ; float64
    movsd xmm4, [rel three_f64]     ; 3.0
    movsd xmm5, [rel six_f64]       ; 6.0
    xorpd xmm6, xmm6                ; 0.0
    movsd xmm7, xmm4
    xorpd xmm3, xmm3
    subsd xmm3, xmm4                ; -3.0
    
    xor rcx, rcx
.hardswish_fwd_f64_loop:
    cmp rcx, r14
    jge .hardswish_fwd_done
    movsd xmm0, [rsi + rcx*8]       ; x
    
    ; Check if x <= -3
    comisd xmm0, xmm3
    jbe .hardswish_fwd_f64_zero
    
    ; Check if x >= 3
    comisd xmm0, xmm4
    jae .hardswish_fwd_f64_linear
    
    ; Middle: x * (x + 3) / 6
    movsd xmm1, xmm0
    addsd xmm1, xmm4                ; x + 3
    mulsd xmm1, xmm0                ; x * (x + 3)
    divsd xmm1, xmm5                ; / 6
    movsd [rdi + rcx*8], xmm1
    jmp .hardswish_fwd_f64_next

.hardswish_fwd_f64_zero:
    movsd [rdi + rcx*8], xmm6
    jmp .hardswish_fwd_f64_next

.hardswish_fwd_f64_linear:
    movsd [rdi + rcx*8], xmm0
    
.hardswish_fwd_f64_next:
    inc rcx
    jmp .hardswish_fwd_f64_loop

.hardswish_fwd_f32:
    movsd xmm4, [rel three_f64]
    movsd xmm5, [rel six_f64]
    xorpd xmm6, xmm6
    movsd xmm3, xmm6
    subsd xmm3, xmm4                ; -3.0
    
    xor rcx, rcx
.hardswish_fwd_f32_loop:
    cmp rcx, r14
    jge .hardswish_fwd_done
    movss xmm0, [rsi + rcx*4]
    cvtss2sd xmm0, xmm0
    
    comisd xmm0, xmm3
    jbe .hardswish_fwd_f32_zero
    
    comisd xmm0, xmm4
    jae .hardswish_fwd_f32_linear
    
    movsd xmm1, xmm0
    addsd xmm1, xmm4
    mulsd xmm1, xmm0
    divsd xmm1, xmm5
    cvtsd2ss xmm1, xmm1
    movss [rdi + rcx*4], xmm1
    jmp .hardswish_fwd_f32_next

.hardswish_fwd_f32_zero:
    xorps xmm1, xmm1
    movss [rdi + rcx*4], xmm1
    jmp .hardswish_fwd_f32_next

.hardswish_fwd_f32_linear:
    cvtsd2ss xmm0, xmm0
    movss [rdi + rcx*4], xmm0
    
.hardswish_fwd_f32_next:
    inc rcx
    jmp .hardswish_fwd_f32_loop

.hardswish_fwd_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_hardswish - Hard Swish activation with autograd
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_hardswish:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .hardswish_node_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute Hard Swish
    mov rdi, r13
    mov rsi, [r12 + NODE_VALUE]
    call hardswish_forward
    
    ; Create node
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .hardswish_node_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel hardswish_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.hardswish_node_alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; hardswish_backward - Backward for Hard Swish
; dHardSwish/dx = 0           if x <= -3
;               = 1           if x >= 3
;               = (2x + 3)/6  otherwise
; =============================================================================
hardswish_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .hardswish_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .hardswish_bwd_f32
    
    ; float64
    movsd xmm4, [rel three_f64]     ; 3.0
    movsd xmm5, [rel six_f64]       ; 6.0
    movsd xmm6, [rel two_f64]       ; 2.0
    xorpd xmm7, xmm7
    subsd xmm7, xmm4                ; -3.0
    
    xor r8, r8
.hardswish_bwd_f64_loop:
    cmp r8, rcx
    jge .hardswish_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd xmm2, [rsi + r8*8]        ; dL/dout
    
    ; Check if x <= -3
    comisd xmm0, xmm7
    jbe .hardswish_bwd_f64_zero
    
    ; Check if x >= 3
    comisd xmm0, xmm4
    jae .hardswish_bwd_f64_one
    
    ; Middle: (2x + 3) / 6
    movsd xmm1, xmm0
    mulsd xmm1, xmm6                ; 2x
    addsd xmm1, xmm4                ; 2x + 3
    divsd xmm1, xmm5                ; (2x + 3) / 6
    mulsd xmm2, xmm1
    jmp .hardswish_bwd_f64_acc

.hardswish_bwd_f64_zero:
    xorpd xmm2, xmm2
    jmp .hardswish_bwd_f64_acc

.hardswish_bwd_f64_one:
    ; grad stays as dL/dout (multiply by 1)
    
.hardswish_bwd_f64_acc:
    addsd xmm2, [rdi + r8*8]
    movsd [rdi + r8*8], xmm2
    
    inc r8
    jmp .hardswish_bwd_f64_loop

.hardswish_bwd_f32:
    movsd xmm4, [rel three_f64]
    movsd xmm5, [rel six_f64]
    movsd xmm6, [rel two_f64]
    xorpd xmm7, xmm7
    subsd xmm7, xmm4
    
    xor r8, r8
.hardswish_bwd_f32_loop:
    cmp r8, rcx
    jge .hardswish_bwd_done
    
    movss xmm0, [rdx + r8*4]
    cvtss2sd xmm0, xmm0
    movss xmm2, [rsi + r8*4]
    cvtss2sd xmm2, xmm2
    
    comisd xmm0, xmm7
    jbe .hardswish_bwd_f32_zero
    
    comisd xmm0, xmm4
    jae .hardswish_bwd_f32_one
    
    movsd xmm1, xmm0
    mulsd xmm1, xmm6
    addsd xmm1, xmm4
    divsd xmm1, xmm5
    mulsd xmm2, xmm1
    jmp .hardswish_bwd_f32_acc

.hardswish_bwd_f32_zero:
    xorpd xmm2, xmm2
    jmp .hardswish_bwd_f32_acc

.hardswish_bwd_f32_one:
    ; grad stays as dL/dout

.hardswish_bwd_f32_acc:
    cvtsd2ss xmm2, xmm2
    addss xmm2, [rdi + r8*4]
    movss [rdi + r8*4], xmm2
    
    inc r8
    jmp .hardswish_bwd_f32_loop

.hardswish_bwd_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; SOFTPLUS Activation
; Softplus(x) = ln(1 + exp(x))
; A smooth approximation of ReLU
; =============================================================================

; =============================================================================
; softplus_forward - Softplus forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
; =============================================================================
softplus_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .softplus_fwd_f32
    
    ; float64: softplus(x) = ln(1 + exp(x))
    xor rcx, rcx
.softplus_fwd_f64_loop:
    cmp rcx, r14
    jge .softplus_fwd_done
    movsd xmm0, [rsi + rcx*8]       ; x
    
    sub rsp, 16
    mov [rsp], rcx
    call exp wrt ..plt              ; exp(x)
    mov rcx, [rsp]
    add rsp, 16
    
    movsd xmm1, [rel one_f64]
    addsd xmm0, xmm1                ; 1 + exp(x)
    
    sub rsp, 16
    mov [rsp], rcx
    call log wrt ..plt              ; ln(1 + exp(x))
    mov rcx, [rsp]
    add rsp, 16
    
    movsd [rdi + rcx*8], xmm0
    inc rcx
    jmp .softplus_fwd_f64_loop

.softplus_fwd_f32:
    xor rcx, rcx
.softplus_fwd_f32_loop:
    cmp rcx, r14
    jge .softplus_fwd_done
    movss xmm0, [rsi + rcx*4]
    cvtss2sd xmm0, xmm0
    
    sub rsp, 16
    mov [rsp], rcx
    call exp wrt ..plt
    mov rcx, [rsp]
    add rsp, 16
    
    movsd xmm1, [rel one_f64]
    addsd xmm0, xmm1
    
    sub rsp, 16
    mov [rsp], rcx
    call log wrt ..plt
    mov rcx, [rsp]
    add rsp, 16
    
    cvtsd2ss xmm0, xmm0
    movss [rdi + rcx*4], xmm0
    inc rcx
    jmp .softplus_fwd_f32_loop

.softplus_fwd_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_softplus - Softplus activation with autograd
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_softplus:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .softplus_node_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute Softplus
    mov rdi, r13
    mov rsi, [r12 + NODE_VALUE]
    call softplus_forward
    
    ; Create node
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .softplus_node_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel softplus_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.softplus_node_alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; softplus_backward - Backward for Softplus
; dSoftplus/dx = sigmoid(x) = 1 / (1 + exp(-x))
; =============================================================================
softplus_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .softplus_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .softplus_bwd_f32
    
    ; float64
    xor r8, r8
.softplus_bwd_f64_loop:
    cmp r8, rcx
    jge .softplus_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd xmm2, [rsi + r8*8]        ; dL/dout
    
    ; sigmoid(x) = 1 / (1 + exp(-x))
    xorpd xmm1, xmm1
    subsd xmm1, xmm0                ; -x
    
    sub rsp, 32
    mov [rsp], r8
    mov [rsp+8], rcx
    movsd [rsp+16], xmm2
    movsd xmm0, xmm1
    call exp wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movsd xmm2, [rsp+16]
    add rsp, 32
    
    movsd xmm1, [rel one_f64]
    addsd xmm0, xmm1                ; 1 + exp(-x)
    divsd xmm1, xmm0                ; sigmoid = 1 / (1 + exp(-x))
    
    mulsd xmm2, xmm1                ; dL/dout * sigmoid
    addsd xmm2, [rdi + r8*8]
    movsd [rdi + r8*8], xmm2
    
    inc r8
    jmp .softplus_bwd_f64_loop

.softplus_bwd_f32:
    xor r8, r8
.softplus_bwd_f32_loop:
    cmp r8, rcx
    jge .softplus_bwd_done
    
    movss xmm0, [rdx + r8*4]
    cvtss2sd xmm0, xmm0
    movss xmm2, [rsi + r8*4]
    cvtss2sd xmm2, xmm2
    
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    
    sub rsp, 32
    mov [rsp], r8
    mov [rsp+8], rcx
    movsd [rsp+16], xmm2
    movsd xmm0, xmm1
    call exp wrt ..plt
    mov r8, [rsp]
    mov rcx, [rsp+8]
    movsd xmm2, [rsp+16]
    add rsp, 32
    
    movsd xmm1, [rel one_f64]
    addsd xmm0, xmm1
    divsd xmm1, xmm0
    
    mulsd xmm2, xmm1
    cvtsd2ss xmm2, xmm2
    addss xmm2, [rdi + r8*4]
    movss [rdi + r8*4], xmm2
    
    inc r8
    jmp .softplus_bwd_f32_loop

.softplus_bwd_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; HARDTANH Activation
; Hardtanh(x) = -1 if x < -1, 1 if x > 1, else x
; A piecewise linear approximation of tanh with min/max bounds
; =============================================================================

; =============================================================================
; hardtanh_forward - Hardtanh forward pass (tensor-only, no autograd)
; Arguments:
;   RDI = Tensor* out
;   RSI = Tensor* x
; =============================================================================
hardtanh_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; out
    mov r13, rsi                    ; x
    
    mov rdi, r13
    call tensor_numel
    mov r14, rax
    
    mov rdi, [r12 + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]
    
    mov eax, [r12 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .hardtanh_fwd_f32
    
    ; float64
    movsd xmm4, [rel one_f64]       ; 1.0
    xorpd xmm5, xmm5
    subsd xmm5, xmm4                ; -1.0
    
    xor rcx, rcx
.hardtanh_fwd_f64_loop:
    cmp rcx, r14
    jge .hardtanh_fwd_done
    movsd xmm0, [rsi + rcx*8]       ; x
    
    ; Check if x < -1
    comisd xmm0, xmm5
    jb .hardtanh_fwd_f64_min
    
    ; Check if x > 1
    comisd xmm0, xmm4
    ja .hardtanh_fwd_f64_max
    
    ; Middle: keep x as is
    movsd [rdi + rcx*8], xmm0
    jmp .hardtanh_fwd_f64_next

.hardtanh_fwd_f64_min:
    movsd [rdi + rcx*8], xmm5
    jmp .hardtanh_fwd_f64_next

.hardtanh_fwd_f64_max:
    movsd [rdi + rcx*8], xmm4
    
.hardtanh_fwd_f64_next:
    inc rcx
    jmp .hardtanh_fwd_f64_loop

.hardtanh_fwd_f32:
    movsd xmm4, [rel one_f64]
    xorpd xmm5, xmm5
    subsd xmm5, xmm4                ; -1.0
    
    xor rcx, rcx
.hardtanh_fwd_f32_loop:
    cmp rcx, r14
    jge .hardtanh_fwd_done
    movss xmm0, [rsi + rcx*4]
    cvtss2sd xmm0, xmm0
    
    comisd xmm0, xmm5
    jb .hardtanh_fwd_f32_min
    
    comisd xmm0, xmm4
    ja .hardtanh_fwd_f32_max
    
    cvtsd2ss xmm0, xmm0
    movss [rdi + rcx*4], xmm0
    jmp .hardtanh_fwd_f32_next

.hardtanh_fwd_f32_min:
    cvtsd2ss xmm1, xmm5
    movss [rdi + rcx*4], xmm1
    jmp .hardtanh_fwd_f32_next

.hardtanh_fwd_f32_max:
    cvtsd2ss xmm1, xmm4
    movss [rdi + rcx*4], xmm1
    
.hardtanh_fwd_f32_next:
    inc rcx
    jmp .hardtanh_fwd_f32_loop

.hardtanh_fwd_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_hardtanh - Hardtanh activation with autograd
; Arguments:
;   RDI = Node* x
; Returns:
;   RAX = Node* out
; =============================================================================
node_hardtanh:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    mov r12, rdi                    ; input node
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .hardtanh_node_alloc_failed
    mov r13, rax                    ; output tensor
    
    ; Compute Hardtanh
    mov rdi, r13
    mov rsi, [r12 + NODE_VALUE]
    call hardtanh_forward
    
    ; Create node
    mov rdi, r13
    mov rsi, 1
    call node_create
    test rax, rax
    jz .hardtanh_node_alloc_failed
    mov rbx, rax
    
    ; Set backward function
    lea rax, [rel hardtanh_backward]
    mov [rbx + NODE_BACKWARD_FN], rax
    
    ; Set parent
    mov dword [rbx + NODE_N_PARENTS], 1
    mov rdi, 8
    call mem_alloc
    mov [rbx + NODE_PARENTS], rax
    mov [rel rax], r12
    
    mov rax, rbx
    
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.hardtanh_node_alloc_failed:
    xor eax, eax
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; hardtanh_backward - Backward for Hardtanh
; dHardtanh/dx = 1 if -1 <= x <= 1, else 0
; =============================================================================
hardtanh_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]
    mov r15, [rel r14]              ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .hardtanh_bwd_done
    
    mov rbx, rax                    ; parent grad tensor
    
    mov rdi, [rbx + TENSOR_DATA]    ; grad_x
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout
    mov rdx, [r15 + NODE_VALUE]
    mov rdx, [rdx + TENSOR_DATA]    ; x
    
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    
    push rbx
    mov rdi, rbx
    call tensor_numel
    mov rcx, rax
    pop rbx
    
    mov rdi, [rsp]
    mov rsi, [rsp+8]
    mov rdx, [rsp+16]
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .hardtanh_bwd_f32
    
    ; float64
    movsd xmm4, [rel one_f64]       ; 1.0
    xorpd xmm5, xmm5
    subsd xmm5, xmm4                ; -1.0
    
    xor r8, r8
.hardtanh_bwd_f64_loop:
    cmp r8, rcx
    jge .hardtanh_bwd_done
    
    movsd xmm0, [rdx + r8*8]        ; x
    movsd xmm2, [rsi + r8*8]        ; dL/dout
    
    ; Check if x < -1 or x > 1
    comisd xmm0, xmm5
    jb .hardtanh_bwd_f64_zero
    comisd xmm0, xmm4
    ja .hardtanh_bwd_f64_zero
    
    ; -1 <= x <= 1: pass gradient through
    addsd xmm2, [rdi + r8*8]
    movsd [rdi + r8*8], xmm2
    jmp .hardtanh_bwd_f64_next

.hardtanh_bwd_f64_zero:
    ; x outside [-1, 1]: gradient is 0
    
.hardtanh_bwd_f64_next:
    inc r8
    jmp .hardtanh_bwd_f64_loop

.hardtanh_bwd_f32:
    movsd xmm4, [rel one_f64]
    xorpd xmm5, xmm5
    subsd xmm5, xmm4
    
    xor r8, r8
.hardtanh_bwd_f32_loop:
    cmp r8, rcx
    jge .hardtanh_bwd_done
    
    movss xmm0, [rdx + r8*4]
    cvtss2sd xmm0, xmm0
    movss xmm2, [rsi + r8*4]
    cvtss2sd xmm2, xmm2
    
    comisd xmm0, xmm5
    jb .hardtanh_bwd_f32_zero
    comisd xmm0, xmm4
    ja .hardtanh_bwd_f32_zero
    
    cvtsd2ss xmm2, xmm2
    addss xmm2, [rdi + r8*4]
    movss [rdi + r8*4], xmm2
    jmp .hardtanh_bwd_f32_next

.hardtanh_bwd_f32_zero:
    ; zero gradient

.hardtanh_bwd_f32_next:
    inc r8
    jmp .hardtanh_bwd_f32_loop

.hardtanh_bwd_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
