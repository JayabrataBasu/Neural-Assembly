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
global relu_backward
global sigmoid_backward
global tanh_backward
global softmax_backward

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
    mov [rax], r12
    
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
    mov r15, [r14]                  ; parent x
    
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
    
    movsd xmm0, [rdx + r8*8]        ; x[i]
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
    mov [rax], r12
    
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
    mov r15, [r15]                  ; parent x
    
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
    mov [rax], r12
    
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
    mov r15, [r15]
    
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
    mov [rsp], rsi                  ; axis
    
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
    mov r14, [rcx]                  ; batch_size
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
    
    movsd xmm0, [rsi]               ; max = in[0]
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
    xor r8, r8
.exp_sum_f64:
    cmp r8, r15
    jge .normalize_f64
    
    push rdi
    push rsi
    push r8
    push rbx
    movsd xmm0, [rsi + r8*8]
    subsd xmm0, [rsp+64]            ; x - max (adjusted for pushes)
    sub rsp, 8
    call exp wrt ..plt
    add rsp, 8
    pop rbx
    pop r8
    pop rsi
    pop rdi
    
    movsd [rdi + r8*8], xmm0        ; out[i] = exp(x - max)
    addsd xmm2, xmm0                ; sum += exp
    movsd [rsp+32], xmm2            ; save sum
    
    inc r8
    jmp .exp_sum_f64

.normalize_f64:
    movsd xmm2, [rsp+32]            ; get sum back
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
    
    movss xmm0, [rsi]
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
    mov [rax], r12
    
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
    mov r15, [r15]                  ; parent x
    
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .done
    
    mov rbx, rax                    ; grad_x tensor
    
    ; Get shape (assume 2D)
    mov rax, [r15 + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov rcx, [rax]                  ; batch_size
    mov [rsp], rcx
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
    cmp r8, [rsp]
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
    cmp r8, [rsp]
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
