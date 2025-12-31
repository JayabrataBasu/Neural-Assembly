; =============================================================================
; nn_layers.asm - Neural Network Layers
; =============================================================================
; Linear, Conv2d modules with forward/backward
; =============================================================================

; Module struct layout (64 bytes):
; Offset  Size    Field
; 0       4       n_params     (uint32_t)
; 4       4       padding
; 8       8       params       (Tensor**)
; 16      8       param_nodes  (Node**)
; 24      8       forward_fn   (void (*)(Module*, Node*, Node**))
; 32      8       config       (void*) - layer-specific config
; 40      24      reserved

%define MODULE_SIZE         64
%define MODULE_N_PARAMS     0
%define MODULE_PARAMS       8
%define MODULE_PARAM_NODES  16
%define MODULE_FORWARD_FN   24
%define MODULE_CONFIG       32

; Linear config struct:
; Offset  Size    Field
; 0       8       in_features
; 8       8       out_features

%define LINEAR_IN_FEATURES  0
%define LINEAR_OUT_FEATURES 8

; Conv2d config struct:
; Offset  Size    Field
; 0       8       in_channels
; 8       8       out_channels
; 16      8       kernel_h
; 24      8       kernel_w
; 32      8       stride
; 40      8       padding

; Activation config struct:
; Offset  Size    Field
; 0       4       activation_type
; 4       4       padding
; 8       8       alpha (for leaky_relu, elu, etc.)

%define ACTIVATION_TYPE     0
%define ACTIVATION_ALPHA    8

; Activation type constants
%define ACT_NONE            0
%define ACT_RELU            1
%define ACT_SIGMOID         2
%define ACT_TANH            3
%define ACT_SOFTMAX         4
%define ACT_GELU            5
%define ACT_LEAKY_RELU      6
%define ACT_ELU             7
%define ACT_SELU            8
%define ACT_SWISH           9
%define ACT_MISH            10
%define ACT_HARDSWISH       11
%define ACT_SOFTPLUS        12
%define ACT_HARDTANH        13

%define NODE_VALUE          0
%define NODE_GRAD           8
%define NODE_BACKWARD_FN    16
%define NODE_N_PARENTS      24
%define NODE_VISITED        28
%define NODE_PARENTS        32
%define NODE_SAVED_TENSORS  40
%define NODE_REQUIRES_GRAD  56

; Node flags for requires_grad field
%define NODE_FLAG_REQUIRES_GRAD  1
%define NODE_FLAG_PERSISTENT     2

%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_DTYPE        32

%define DT_FLOAT32          0
%define DT_FLOAT64          1

section .data
    align 8
    err_null_module:    db "Error: Null module", 0
    
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
extern tensor_fill
extern node_create
extern node_matmul
extern node_add
extern ew_add
extern ew_mul
extern matmul
extern tensor_transpose_2d
extern panic

; For random initialization
extern rand
extern srand
extern time

; Linear layer functions
extern neural_linear_free
extern neural_linear_forward
extern neural_linear_weight
extern neural_linear_bias

; Activation functions from autograd
extern node_relu
extern node_sigmoid
extern node_tanh
extern node_softmax
extern node_gelu
extern node_leaky_relu
extern node_elu
extern node_selu
extern node_swish
extern node_mish
extern node_hardswish
extern node_softplus
extern node_hardtanh

; Export layer functions
global linear_create
global linear_forward
global linear_forward_fn
global linear_backward
global conv2d_create
global conv2d_forward
global conv2d_forward_fn
global conv2d_backward

; Export activation layer functions
global activation_create
global activation_forward_fn
global activation_relu_create
global activation_sigmoid_create
global activation_tanh_create
global activation_softmax_create
global activation_gelu_create
global activation_leaky_relu_create
global activation_elu_create
global activation_selu_create
global activation_swish_create
global activation_mish_create
global activation_hardswish_create
global activation_softplus_create
global activation_hardtanh_create
global module_free
global module_get_params

; =============================================================================
; Helper: Xavier/Glorot initialization
; Arguments:
;   RDI = Tensor* tensor
;   RSI = fan_in (uint64_t)
;   RDX = fan_out (uint64_t)
; =============================================================================
xavier_init:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; tensor
    mov r13, rsi                    ; fan_in
    mov r14, rdx                    ; fan_out
    
    ; Scale = sqrt(6 / (fan_in + fan_out))
    ; For simplicity, use scale = sqrt(2 / fan_in) (He init variant)
    
    ; Seed random if needed (simple approach)
    xor edi, edi
    call time wrt ..plt
    mov edi, eax
    call srand wrt ..plt
    
    ; Get element count
    mov rdi, r12
    call tensor_numel
    mov r15, rax
    
    ; Calculate scale: sqrt(2.0 / fan_in)
    cvtsi2sd xmm0, r13              ; fan_in as double
    mov rax, 2
    cvtsi2sd xmm1, rax
    divsd xmm1, xmm0                ; 2.0 / fan_in
    sqrtsd xmm1, xmm1               ; sqrt
    movsd [rel rsp], xmm1               ; save scale
    
    mov rbx, [r12 + TENSOR_DATA]
    mov eax, [r12 + TENSOR_DTYPE]
    mov [rsp+8], eax
    
    xor rcx, rcx
.init_loop:
    cmp rcx, r15
    jge .done
    
    push rcx
    ; Generate random value in [-1, 1]
    call rand wrt ..plt
    ; Convert to double in [0, 1]
    cvtsi2sd xmm0, eax
    mov eax, 0x7FFFFFFF             ; RAND_MAX approx
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1                ; [0, 1]
    
    ; Scale to [-1, 1]
    addsd xmm0, xmm0                ; [0, 2]
    mov rax, 1
    cvtsi2sd xmm1, rax
    subsd xmm0, xmm1                ; [-1, 1]
    
    ; Apply scale
    mulsd xmm0, [rsp+8]             ; Note: scale is at [rel rsp] but we pushed rcx
    
    pop rcx
    
    mov eax, [rsp+8]
    cmp eax, DT_FLOAT32
    je .store_f32
    
    movsd [rbx + rcx*8], xmm0
    jmp .next

.store_f32:
    cvtsd2ss xmm0, xmm0
    movss [rbx + rcx*4], xmm0

.next:
    inc rcx
    jmp .init_loop

.done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; linear_create - Create a Linear layer
; Arguments:
;   RDI = in_features (uint64_t)
;   RSI = out_features (uint64_t)
;   RDX = dtype (uint32_t)
; Returns:
;   RAX = Module*
; =============================================================================
linear_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; in_features
    mov r13, rsi                    ; out_features
    mov r14d, edx                   ; dtype
    
    ; Allocate module struct
    mov rdi, MODULE_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed
    mov r15, rax                    ; module
    
    ; Initialize module
    mov dword [r15 + MODULE_N_PARAMS], 2    ; weight and bias
    
    ; Allocate config
    mov rdi, 16                     ; 2 uint64_t
    call mem_alloc
    mov [r15 + MODULE_CONFIG], rax
    mov [rax + LINEAR_IN_FEATURES], r12
    mov [rax + LINEAR_OUT_FEATURES], r13
    
    ; Allocate params array (2 Tensor*)
    mov rdi, 16
    call mem_alloc
    mov [r15 + MODULE_PARAMS], rax
    mov rbx, rax
    
    ; Create weight tensor (out_features x in_features)
    mov [rel rsp], r13                  ; shape[0] = out_features
    mov [rsp+8], r12                ; shape[1] = in_features
    mov rdi, 2
    lea rsi, [rel rsp]
    mov edx, r14d
    call tensor_create
    mov [rel rbx], rax                  ; params[0] = weight
    mov [rsp+16], rax
    
    ; Initialize weight with Xavier
    mov rdi, rax
    mov rsi, r12                    ; fan_in
    mov rdx, r13                    ; fan_out
    call xavier_init
    
    ; Create bias tensor (out_features)
    mov [rel rsp], r13                  ; shape[0] = out_features
    mov rdi, 1
    lea rsi, [rel rsp]
    mov edx, r14d
    call tensor_create
    mov [rbx + 8], rax              ; params[1] = bias
    ; Bias initialized to zeros (default)
    
    ; Allocate param_nodes array
    mov rdi, 16
    call mem_alloc
    mov [r15 + MODULE_PARAM_NODES], rax
    mov [rsp+24], rax
    
    ; Create nodes for parameters (with PERSISTENT flag to prevent accidental freeing)
    mov rdi, [rel rbx]                  ; weight tensor
    mov rsi, (NODE_FLAG_REQUIRES_GRAD | NODE_FLAG_PERSISTENT)  ; requires_grad + persistent
    call node_create
    mov rcx, [rsp+24]
    mov [rel rcx], rax                  ; param_nodes[0]
    
    mov rdi, [rbx + 8]              ; bias tensor
    mov rsi, (NODE_FLAG_REQUIRES_GRAD | NODE_FLAG_PERSISTENT)  ; requires_grad + persistent
    call node_create
    mov rcx, [rsp+24]
    mov [rcx + 8], rax              ; param_nodes[1]
    
    ; Set forward function
    lea rax, [rel linear_forward_fn]
    mov [r15 + MODULE_FORWARD_FN], rax
    
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
; linear_forward_fn - Forward pass for Linear layer (called via function pointer)
; Arguments:
;   RDI = Module* self
;   RSI = Node* input (batch_size x in_features)
;   RDX = Node** output (pointer to store output Node*)
; =============================================================================
linear_forward_fn:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 72
    
    mov r12, rdi                    ; module
    mov r13, rsi                    ; input node
    mov r14, rdx                    ; output pointer
    
    ; Get config
    mov rax, [r12 + MODULE_CONFIG]
    mov r15, [rax + LINEAR_IN_FEATURES]     ; in_features
    mov rbx, [rax + LINEAR_OUT_FEATURES]    ; out_features
    
    ; Get weight and bias nodes
    mov rax, [r12 + MODULE_PARAM_NODES]
    mov rcx, [rel rax]                  ; weight_node
    mov [rel rsp], rcx
    mov rcx, [rax + 8]              ; bias_node
    mov [rsp+8], rcx
    
    ; Forward: y = x @ W^T + b
    ; First create W^T
    mov rax, [r12 + MODULE_PARAMS]
    mov rdi, [rel rax]                  ; weight tensor (out x in)
    mov [rsp+16], rdi
    
    ; Create transposed weight tensor (in x out)
    mov qword [rsp+24], 0
    mov [rsp+24], r15               ; shape[0] = in_features
    mov [rsp+32], rbx               ; shape[1] = out_features
    
    mov rax, [rsp+16]
    mov rdi, 2
    lea rsi, [rsp+24]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+40], rax               ; W^T tensor
    
    ; Transpose weight
    mov rdi, rax
    mov rsi, [rsp+16]
    call tensor_transpose_2d
    
    ; Create node for W^T (temporary, shares the grad with weight)
    mov rdi, [rsp+40]
    mov rsi, 1
    call node_create
    mov [rsp+48], rax               ; W^T node
    
    ; Compute x @ W^T
    mov rdi, r13                    ; input node
    mov rsi, [rsp+48]               ; W^T node
    call node_matmul
    mov [rsp+56], rax               ; matmul result node
    
    ; Get batch size from input
    mov rax, [r13 + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov rcx, [rel rax]                  ; batch_size
    mov [rsp+64], rcx
    
    ; Create output tensor for bias addition
    mov [rsp+24], rcx               ; batch_size
    mov [rsp+32], rbx               ; out_features
    
    mov rax, [rsp+56]
    mov rax, [rax + NODE_VALUE]
    mov rdi, 2
    lea rsi, [rsp+24]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov r15, rax                    ; output tensor
    
    ; Add bias to each row: out[i,:] = matmul[i,:] + bias
    ; For simplicity, loop over batch
    mov rdi, [r15 + TENSOR_DATA]
    mov rax, [rsp+56]
    mov rax, [rax + NODE_VALUE]
    mov rsi, [rax + TENSOR_DATA]
    mov rax, [rsp+8]                ; bias node
    mov rax, [rax + NODE_VALUE]
    mov rdx, [rax + TENSOR_DATA]    ; bias data
    
    mov rcx, [rsp+64]               ; batch_size
    mov r8, rbx                     ; out_features (saved earlier as rbx)
    
    mov eax, [r15 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .add_bias_f32
    
    ; float64
    xor r9, r9                      ; batch idx
.add_bias_f64_batch:
    cmp r9, rcx
    jge .create_output_node
    
    xor r10, r10                    ; feature idx
.add_bias_f64_feat:
    cmp r10, r8
    jge .next_batch_f64
    
    mov rax, r9
    imul rax, r8
    add rax, r10
    
    movsd xmm0, [rsi + rax*8]
    addsd xmm0, [rdx + r10*8]
    movsd [rdi + rax*8], xmm0
    
    inc r10
    jmp .add_bias_f64_feat

.next_batch_f64:
    inc r9
    jmp .add_bias_f64_batch

.add_bias_f32:
    xor r9, r9
.add_bias_f32_batch:
    cmp r9, rcx
    jge .create_output_node
    
    xor r10, r10
.add_bias_f32_feat:
    cmp r10, r8
    jge .next_batch_f32
    
    mov rax, r9
    imul rax, r8
    add rax, r10
    
    movss xmm0, [rsi + rax*4]
    addss xmm0, [rdx + r10*4]
    movss [rdi + rax*4], xmm0
    
    inc r10
    jmp .add_bias_f32_feat

.next_batch_f32:
    inc r9
    jmp .add_bias_f32_batch

.create_output_node:
    mov rdi, r15
    mov rsi, 1
    call node_create
    
    ; Set backward function
    lea rcx, [rel linear_backward]
    mov [rax + NODE_BACKWARD_FN], rcx
    
    ; Set parents (input, weight_node, bias_node)
    mov dword [rax + NODE_N_PARENTS], 3
    push rax
    mov rdi, 24
    call mem_alloc
    pop rcx
    mov [rcx + NODE_PARENTS], rax
    mov [rel rax], r13                  ; parent[0] = input
    mov rdx, [rel rsp]
    mov [rax + 8], rdx              ; parent[1] = weight
    mov rdx, [rsp+8]
    mov [rax + 16], rdx             ; parent[2] = bias
    
    ; Save module reference for backward
    mov [rcx + NODE_SAVED_TENSORS], r12
    
    ; Store output
    mov [rel r14], rcx
    
    ; Cleanup temporary W^T tensor (but not the node, it's in the graph)
    ; Actually, we should keep it for backward...
    ; For now, leave it
    
    add rsp, 72
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; linear_forward - Wrapper for linear forward
; Arguments:
;   RDI = Module* self
;   RSI = Node* input
; Returns:
;   RAX = Node* output
; =============================================================================
linear_forward:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    ; RDI = module, RSI = input, RDX = &output
    lea rdx, [rbp - 8]              ; output pointer
    call linear_forward_fn
    
    mov rax, [rbp - 8]              ; return output node
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; linear_backward - Backward for Linear layer
; dL/dx = dL/dout @ W
; dL/dW = x^T @ dL/dout
; dL/db = sum(dL/dout, axis=0)
; =============================================================================
linear_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 88
    
    mov r12, rdi                    ; self node
    mov r13, [r12 + NODE_GRAD]      ; dL/dout (batch x out_features)
    mov r14, [r12 + NODE_PARENTS]   ; parents array
    
    ; Get parents
    mov rax, [rel r14]                  ; input node
    mov [rel rsp], rax
    mov rax, [r14 + 8]              ; weight node
    mov [rsp+8], rax
    mov rax, [r14 + 16]             ; bias node
    mov [rsp+16], rax
    
    ; Get dimensions from dL/dout
    mov rax, [r13 + TENSOR_SHAPE]
    mov rcx, [rel rax]                  ; batch_size
    mov [rsp+24], rcx
    mov rcx, [rax + 8]              ; out_features
    mov [rsp+32], rcx
    
    ; Get input shape
    mov rax, [rel rsp]
    mov rax, [rax + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov rcx, [rax + 8]              ; in_features
    mov [rsp+40], rcx
    
    ; 1. dL/db = sum over batch dimension
    mov rax, [rsp+16]               ; bias node
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .skip_bias_grad
    
    mov rbx, rdi                    ; bias grad tensor
    mov rdi, [rbx + TENSOR_DATA]
    mov rsi, [r13 + TENSOR_DATA]    ; dL/dout data
    
    mov rcx, [rsp+24]               ; batch_size
    mov r8, [rsp+32]                ; out_features
    
    mov eax, [rbx + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .bias_grad_f32
    
    ; float64: sum over batch for each output feature
    xor r9, r9                      ; feature idx
.bias_grad_f64_feat:
    cmp r9, r8
    jge .skip_bias_grad
    
    vxorpd xmm0, xmm0, xmm0
    xor r10, r10                    ; batch idx
.bias_grad_f64_batch:
    cmp r10, rcx
    jge .store_bias_f64
    
    mov rax, r10
    imul rax, r8
    add rax, r9
    addsd xmm0, [rsi + rax*8]
    
    inc r10
    jmp .bias_grad_f64_batch

.store_bias_f64:
    addsd xmm0, [rdi + r9*8]
    movsd [rdi + r9*8], xmm0
    inc r9
    jmp .bias_grad_f64_feat

.bias_grad_f32:
    xor r9, r9
.bias_grad_f32_feat:
    cmp r9, r8
    jge .skip_bias_grad
    
    vxorps xmm0, xmm0, xmm0
    xor r10, r10
.bias_grad_f32_batch:
    cmp r10, rcx
    jge .store_bias_f32
    
    mov rax, r10
    imul rax, r8
    add rax, r9
    addss xmm0, [rsi + rax*4]
    
    inc r10
    jmp .bias_grad_f32_batch

.store_bias_f32:
    addss xmm0, [rdi + r9*4]
    movss [rdi + r9*4], xmm0
    inc r9
    jmp .bias_grad_f32_feat

.skip_bias_grad:
    ; 2. dL/dW = x^T @ dL/dout
    ; dL/dout is (batch x out), x is (batch x in)
    ; x^T is (in x batch), so x^T @ dL/dout is (in x out)
    ; But W is (out x in), so we need dL/dW = dL/dout^T @ x = (out x batch) @ (batch x in) = (out x in)
    
    mov rax, [rsp+8]                ; weight node
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .skip_weight_grad
    
    mov [rsp+48], rdi               ; weight grad (out x in)
    
    ; Create dL/dout^T (out x batch)
    mov rcx, [rsp+32]               ; out_features
    mov [rsp+56], rcx
    mov rcx, [rsp+24]               ; batch_size
    mov [rsp+64], rcx
    
    mov rdi, 2
    lea rsi, [rsp+56]
    mov eax, [r13 + TENSOR_DTYPE]
    mov edx, eax
    call tensor_create
    mov [rsp+72], rax               ; dL/dout^T tensor
    
    ; Transpose dL/dout
    mov rdi, rax
    mov rsi, r13
    call tensor_transpose_2d
    
    ; Create temp for matmul result (out x in)
    mov rcx, [rsp+32]
    mov [rsp+56], rcx
    mov rcx, [rsp+40]
    mov [rsp+64], rcx
    
    mov rdi, 2
    lea rsi, [rsp+56]
    mov rax, [rsp+48]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+80], rax               ; temp tensor
    
    ; matmul: temp = dL/dout^T @ x
    mov rdi, rax
    mov rsi, [rsp+72]               ; dL/dout^T
    mov rax, [rel rsp]                  ; input node
    mov rdx, [rax + NODE_VALUE]     ; x tensor
    call matmul
    
    ; Add to weight grad
    mov rdi, [rsp+48]
    mov rsi, [rsp+48]
    mov rdx, [rsp+80]
    call ew_add
    
    ; Cleanup temps
    mov rdi, [rsp+72]
    call tensor_free
    mov rdi, [rsp+80]
    call tensor_free

.skip_weight_grad:
    ; 3. dL/dx = dL/dout @ W
    mov rax, [rel rsp]                  ; input node
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .done
    
    mov [rsp+48], rdi               ; input grad (batch x in)
    
    ; Create temp for matmul result
    mov rcx, [rsp+24]               ; batch
    mov [rsp+56], rcx
    mov rcx, [rsp+40]               ; in_features
    mov [rsp+64], rcx
    
    mov rdi, 2
    lea rsi, [rsp+56]
    mov rax, [rsp+48]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+72], rax
    
    ; matmul: temp = dL/dout @ W
    mov rdi, rax
    mov rsi, r13                    ; dL/dout
    mov rax, [rsp+8]                ; weight node
    mov rdx, [rax + NODE_VALUE]     ; W tensor
    call matmul
    
    ; Add to input grad
    mov rdi, [rsp+48]
    mov rsi, [rsp+48]
    mov rdx, [rsp+72]
    call ew_add
    
    ; Cleanup
    mov rdi, [rsp+72]
    call tensor_free

.done:
    add rsp, 88
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; Activation Module Implementation
; =============================================================================

; =============================================================================
; activation_create - Create an activation layer module
; Arguments:
;   RDI = activation_type (ACT_RELU, ACT_SIGMOID, etc.)
;   RSI = alpha (for leaky_relu, elu, etc.) - optional, pass 0 for defaults
; Returns:
;   RAX = Module* or NULL on error
; =============================================================================
activation_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 32
    
    mov r12d, edi                   ; activation type
    movsd [rsp], xmm0               ; save alpha (passed in xmm0 for float params)
    mov r13, rsi                    ; alpha as integer (or 0)
    
    ; Allocate module struct
    mov rdi, MODULE_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed
    mov r14, rax                    ; module
    
    ; Initialize module - no params for activation
    mov dword [r14 + MODULE_N_PARAMS], 0
    mov qword [r14 + MODULE_PARAMS], 0
    mov qword [r14 + MODULE_PARAM_NODES], 0
    
    ; Allocate config
    mov rdi, 16                     ; activation config size
    call mem_alloc
    test rax, rax
    jz .alloc_config_failed
    mov [r14 + MODULE_CONFIG], rax
    mov [rax + ACTIVATION_TYPE], r12d
    
    ; Set alpha based on activation type
    cmp r12d, ACT_LEAKY_RELU
    je .set_leaky_alpha
    cmp r12d, ACT_ELU
    je .set_elu_alpha
    ; Default: no alpha needed
    mov qword [rax + ACTIVATION_ALPHA], 0
    jmp .set_forward_fn
    
.set_leaky_alpha:
    ; Default alpha for leaky relu: 0.01
    test r13, r13
    jnz .use_custom_alpha
    mov rdi, 0x3F847AE147AE147B     ; 0.01 in double
    mov [rax + ACTIVATION_ALPHA], rdi
    jmp .set_forward_fn
    
.set_elu_alpha:
    ; Default alpha for ELU: 1.0
    test r13, r13
    jnz .use_custom_alpha
    mov rdi, 0x3FF0000000000000     ; 1.0 in double
    mov [rax + ACTIVATION_ALPHA], rdi
    jmp .set_forward_fn
    
.use_custom_alpha:
    mov [rax + ACTIVATION_ALPHA], r13
    
.set_forward_fn:
    ; Set forward function
    lea rax, [rel activation_forward_fn]
    mov [r14 + MODULE_FORWARD_FN], rax
    
    mov rax, r14
    
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_config_failed:
    mov rdi, r14
    call mem_free
.alloc_failed:
    xor eax, eax
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; activation_forward_fn - Forward pass for activation layer (via function pointer)
; Arguments:
;   RDI = Module* self
;   RSI = Node* input
;   RDX = Node** output (pointer to store output Node*)
; =============================================================================
activation_forward_fn:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 24
    
    mov r12, rdi                    ; module
    mov r13, rsi                    ; input node
    mov rbx, rdx                    ; output pointer
    
    ; Get activation type from config
    mov rax, [r12 + MODULE_CONFIG]
    mov eax, [rax + ACTIVATION_TYPE]
    
    ; Dispatch based on activation type
    cmp eax, ACT_RELU
    je .forward_relu
    cmp eax, ACT_SIGMOID
    je .forward_sigmoid
    cmp eax, ACT_TANH
    je .forward_tanh
    cmp eax, ACT_SOFTMAX
    je .forward_softmax
    cmp eax, ACT_GELU
    je .forward_gelu
    cmp eax, ACT_LEAKY_RELU
    je .forward_leaky_relu
    cmp eax, ACT_ELU
    je .forward_elu
    cmp eax, ACT_SELU
    je .forward_selu
    cmp eax, ACT_SWISH
    je .forward_swish
    cmp eax, ACT_MISH
    je .forward_mish
    cmp eax, ACT_HARDSWISH
    je .forward_hardswish
    cmp eax, ACT_SOFTPLUS
    je .forward_softplus
    cmp eax, ACT_HARDTANH
    je .forward_hardtanh
    
    ; Unknown activation - pass through
    mov [rbx], r13
    xor eax, eax
    jmp .done
    
.forward_relu:
    mov rdi, r13
    call node_relu
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_sigmoid:
    mov rdi, r13
    call node_sigmoid
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_tanh:
    mov rdi, r13
    call node_tanh
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_softmax:
    mov rdi, r13
    call node_softmax
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_gelu:
    mov rdi, r13
    call node_gelu
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_leaky_relu:
    mov rdi, r13
    ; Get alpha from config
    mov rax, [r12 + MODULE_CONFIG]
    movsd xmm0, [rax + ACTIVATION_ALPHA]
    call node_leaky_relu
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_elu:
    mov rdi, r13
    mov rax, [r12 + MODULE_CONFIG]
    movsd xmm0, [rax + ACTIVATION_ALPHA]
    call node_elu
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_selu:
    mov rdi, r13
    call node_selu
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_swish:
    mov rdi, r13
    call node_swish
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_mish:
    mov rdi, r13
    call node_mish
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_hardswish:
    mov rdi, r13
    call node_hardswish
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_softplus:
    mov rdi, r13
    call node_softplus
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.forward_hardtanh:
    mov rdi, r13
    call node_hardtanh
    mov [rbx], rax
    xor eax, eax
    jmp .done
    
.done:
    add rsp, 24
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; Convenience functions to create specific activation layers
; =============================================================================

activation_relu_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_RELU
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_sigmoid_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_SIGMOID
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_tanh_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_TANH
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_softmax_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_SOFTMAX
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_gelu_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_GELU
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_leaky_relu_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_LEAKY_RELU
    xor esi, esi                    ; use default alpha
    call activation_create
    pop rbp
    ret

activation_elu_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_ELU
    xor esi, esi                    ; use default alpha
    call activation_create
    pop rbp
    ret

activation_selu_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_SELU
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_swish_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_SWISH
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_mish_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_MISH
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_hardswish_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_HARDSWISH
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_softplus_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_SOFTPLUS
    xor esi, esi
    call activation_create
    pop rbp
    ret

activation_hardtanh_create:
    push rbp
    mov rbp, rsp
    mov edi, ACT_HARDTANH
    xor esi, esi
    call activation_create
    pop rbp
    ret

; =============================================================================
; conv2d_create - Create a Conv2d layer (placeholder - basic implementation)
; Arguments:
;   RDI = in_channels
;   RSI = out_channels
;   RDX = kernel_h
;   RCX = kernel_w
;   R8 = stride
;   R9 = padding
;   [rel stack] = dtype
; Returns:
;   RAX = Module*
; =============================================================================
conv2d_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 72
    
    ; Save parameters
    mov [rel rsp], rdi                  ; in_channels
    mov [rsp+8], rsi                ; out_channels
    mov [rsp+16], rdx               ; kernel_h
    mov [rsp+24], rcx               ; kernel_w
    mov [rsp+32], r8                ; stride
    mov [rsp+40], r9                ; padding
    mov eax, [rbp+16]               ; dtype from stack
    mov [rsp+48], eax
    
    ; Allocate module
    mov rdi, MODULE_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed
    mov r15, rax
    
    mov dword [r15 + MODULE_N_PARAMS], 2
    
    ; Allocate config (48 bytes for conv2d config)
    mov rdi, 48
    call mem_alloc
    mov [r15 + MODULE_CONFIG], rax
    mov rbx, rax
    
    ; Copy config
    mov rax, [rel rsp]
    mov [rel rbx], rax                  ; in_channels
    mov rax, [rsp+8]
    mov [rbx+8], rax                ; out_channels
    mov rax, [rsp+16]
    mov [rbx+16], rax               ; kernel_h
    mov rax, [rsp+24]
    mov [rbx+24], rax               ; kernel_w
    mov rax, [rsp+32]
    mov [rbx+32], rax               ; stride
    mov rax, [rsp+40]
    mov [rbx+40], rax               ; padding
    
    ; Allocate params array
    mov rdi, 16
    call mem_alloc
    mov [r15 + MODULE_PARAMS], rax
    mov r12, rax
    
    ; Create weight tensor (out_channels x in_channels x kernel_h x kernel_w)
    mov rax, [rsp+8]                ; out_channels
    mov [rsp+52], rax
    mov rax, [rel rsp]                  ; in_channels
    mov [rsp+60], rax
    ; Note: For simplicity, treating as 2D (out x in*kh*kw)
    mov rax, [rel rsp]
    imul rax, [rsp+16]
    imul rax, [rsp+24]              ; in * kh * kw
    mov [rsp+60], rax
    
    mov rdi, 2
    lea rsi, [rsp+52]
    mov edx, [rsp+48]
    call tensor_create
    mov [rel r12], rax
    
    ; Initialize weight
    mov rdi, rax
    mov rsi, [rsp+60]               ; fan_in = in * kh * kw
    mov rdx, [rsp+8]                ; fan_out = out
    call xavier_init
    
    ; Create bias tensor (out_channels)
    mov rax, [rsp+8]
    mov [rsp+52], rax
    mov rdi, 1
    lea rsi, [rsp+52]
    mov edx, [rsp+48]
    call tensor_create
    mov [r12+8], rax
    
    ; Create param nodes
    mov rdi, 16
    call mem_alloc
    mov [r15 + MODULE_PARAM_NODES], rax
    mov r13, rax
    
    mov rdi, [rel r12]
    mov rsi, 1
    call node_create
    mov [rel r13], rax
    
    mov rdi, [r12+8]
    mov rsi, 1
    call node_create
    mov [r13+8], rax
    
    ; Set forward function
    lea rax, [rel conv2d_forward_fn]
    mov [r15 + MODULE_FORWARD_FN], rax
    
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
; conv2d_forward_fn - Forward for Conv2d (naive implementation)
; Arguments:
;   RDI = Module* self
;   RSI = Node* input (batch x in_channels x height x width)
;   RDX = Node** output
; =============================================================================
conv2d_forward_fn:
    ; This is a placeholder - full conv2d is complex
    ; For now, just return input (identity)
    push rbp
    mov rbp, rsp
    
    mov rax, rsi
    mov [rel rdx], rax
    
    pop rbp
    ret

; =============================================================================
; conv2d_forward - Wrapper
; =============================================================================
conv2d_forward:
    jmp conv2d_forward_fn

; =============================================================================
; conv2d_backward - Backward for Conv2d
; =============================================================================
conv2d_backward:
    ; Placeholder
    ret

; =============================================================================
; module_free - Free a module and its parameters
; Arguments:
;   RDI = Module* module
; =============================================================================
module_free:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi
    
    ; Free params
    mov ecx, [r12 + MODULE_N_PARAMS]
    mov rbx, [r12 + MODULE_PARAMS]
    test rbx, rbx
    jz .skip_params
    
    xor r13d, r13d
.free_params_loop:
    cmp r13d, ecx
    jge .free_params_array
    
    mov rdi, [rbx + r13*8]
    test rdi, rdi
    jz .next_param
    push rcx
    call tensor_free
    pop rcx
.next_param:
    inc r13d
    jmp .free_params_loop

.free_params_array:
    mov rdi, rbx
    call mem_free

.skip_params:
    ; Free param_nodes array (nodes should be freed by autograd)
    mov rdi, [r12 + MODULE_PARAM_NODES]
    test rdi, rdi
    jz .skip_nodes
    call mem_free

.skip_nodes:
    ; Free config
    mov rdi, [r12 + MODULE_CONFIG]
    test rdi, rdi
    jz .skip_config
    call mem_free

.skip_config:
    ; Free module struct
    mov rdi, r12
    call mem_free

.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; module_get_params - Get parameter tensors from module
; Arguments:
;   RDI = Module* module
;   RSI = Tensor*** out_params (output: pointer to array)
;   RDX = uint32_t* out_n (output: number of params)
; =============================================================================
module_get_params:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null_module
    
    mov eax, [rdi + MODULE_N_PARAMS]
    mov [rel rdx], eax
    
    mov rax, [rdi + MODULE_PARAMS]
    mov [rel rsi], rax
    
    pop rbp
    ret

.null_module:
    mov dword [rel rdx], 0
    mov qword [rel rsi], 0
    pop rbp
    ret

; =============================================================================
; Sequential Container Implementation
; =============================================================================

; Sequential struct layout:
; Offset  Size    Field
; 0       8       capacity         (uint64_t) - allocated capacity
; 8       8       size             (uint64_t) - number of modules
; 16      8       modules          (NeuralLinear**) - array of module pointers
; 24      8       intermediates    (Node**) - array of intermediate outputs
; 32      8       inter_capacity   (uint64_t) - intermediate array capacity
; 40      1       save_inter       (uint8_t) - flag to save intermediates

%define SEQUENTIAL_CAPACITY       0
%define SEQUENTIAL_SIZE           8
%define SEQUENTIAL_MODULES        16
%define SEQUENTIAL_INTERMEDIATES  24
%define SEQUENTIAL_INTER_CAP      32
%define SEQUENTIAL_SAVE_INTER     40
%define SEQUENTIAL_SIZE_BYTES     48

; Export sequential functions
global neural_sequential_create
global neural_sequential_free
global neural_sequential_add
global neural_sequential_forward
global neural_sequential_size
global neural_sequential_get
global neural_sequential_parameters
global neural_sequential_get_intermediate
global neural_sequential_set_save_intermediates
global neural_sequential_clear_intermediates

; =============================================================================
; neural_sequential_create - Create a sequential container
; Arguments:
;   RDI = NeuralLinear** modules (can be NULL)
;   RSI = uint64_t num_modules
; Returns:
;   RAX = NeuralSequential* or NULL on error
; =============================================================================
neural_sequential_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    ; Save original parameters before mem_alloc clobbers them
    mov r14, rdi                    ; r14 = modules array (may be NULL)
    mov r15, rsi                    ; r15 = num_modules
    
    ; Allocate sequential struct
    mov rdi, SEQUENTIAL_SIZE_BYTES
    call mem_alloc
    test rax, rax
    jz .error
    mov r12, rax                    ; r12 = sequential
    
    ; Initialize
    mov qword [r12 + SEQUENTIAL_SIZE], 0
    mov qword [r12 + SEQUENTIAL_CAPACITY], 0
    mov qword [r12 + SEQUENTIAL_MODULES], 0
    mov qword [r12 + SEQUENTIAL_INTERMEDIATES], 0
    mov qword [r12 + SEQUENTIAL_INTER_CAP], 0
    mov byte [r12 + SEQUENTIAL_SAVE_INTER], 0
    
    ; If modules provided, add them
    test r14, r14
    jz .done
    test r15, r15
    jz .done
    
    mov rbx, r14                    ; rbx = modules array
    mov r13, r15                    ; r13 = num_modules
    
.add_loop:
    mov rdi, r12                    ; sequential
    mov rsi, [rbx]                  ; current module
    call neural_sequential_add
    test eax, eax
    jnz .error_free
    
    add rbx, 8                      ; next module pointer
    dec r13
    jnz .add_loop
    
.done:
    mov rax, r12
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error_free:
    mov rdi, r12
    call neural_sequential_free
.error:
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
; neural_sequential_free - Free a sequential container
; Arguments:
;   RDI = NeuralSequential* seq
; =============================================================================
neural_sequential_free:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi                    ; r12 = sequential
    
    ; Free all modules
    mov rbx, [r12 + SEQUENTIAL_MODULES]
    test rbx, rbx
    jz .free_struct
    
    xor rcx, rcx
.free_loop:
    cmp rcx, [r12 + SEQUENTIAL_SIZE]
    jge .free_array
    
    mov rdi, [rbx + rcx*8]          ; module
    test rdi, rdi
    jz .next_module
    
    ; Call module free function
    push rcx
    push rbx
    call module_free
    pop rbx
    pop rcx
    
.next_module:
    inc rcx
    jmp .free_loop
    
.free_array:
    mov rdi, [r12 + SEQUENTIAL_MODULES]
    call mem_free
    
    ; Free intermediates array if exists
    mov rdi, [r12 + SEQUENTIAL_INTERMEDIATES]
    test rdi, rdi
    jz .free_struct
    call mem_free
    
.free_struct:
    mov rdi, r12
    call mem_free
    
.done:
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_sequential_add - Add a module to sequential container
; Arguments:
;   RDI = NeuralSequential* seq
;   RSI = NeuralLinear* module
; Returns:
;   EAX = 0 on success, error code on failure
; =============================================================================
neural_sequential_add:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    test rdi, rdi
    jz .error_null
    test rsi, rsi
    jz .error_null
    
    mov r12, rdi                    ; r12 = sequential
    mov r13, rsi                    ; r13 = module
    
    ; Check if we need to resize
    mov rax, [r12 + SEQUENTIAL_SIZE]
    cmp rax, [r12 + SEQUENTIAL_CAPACITY]
    jl .add_module
    
    ; Need to resize - double capacity or start with 4
    mov rbx, [r12 + SEQUENTIAL_CAPACITY]
    test rbx, rbx
    jnz .double_capacity
    mov rbx, 4                      ; initial capacity
    jmp .resize
    
.double_capacity:
    shl rbx, 1                      ; double capacity
    
.resize:
    ; Allocate new array
    mov rdi, rbx
    shl rdi, 3                      ; 8 bytes per pointer
    call mem_alloc
    test rax, rax
    jz .error_memory
    
    mov rcx, rax                    ; rcx = new array
    
    ; Copy existing modules
    mov rdx, [r12 + SEQUENTIAL_MODULES]
    test rdx, rdx
    jz .copy_done
    
    mov rsi, [r12 + SEQUENTIAL_SIZE]
    test rsi, rsi
    jz .copy_done
    
    ; Copy old array to new
    push rcx
    mov rdi, rcx
    mov rcx, rsi
    rep movsq
    pop rcx
    
.copy_done:
    ; Free old array
    mov rdi, [r12 + SEQUENTIAL_MODULES]
    test rdi, rdi
    jz .update_struct
    call mem_free
    
.update_struct:
    mov [r12 + SEQUENTIAL_MODULES], rcx
    mov [r12 + SEQUENTIAL_CAPACITY], rbx
    
.add_module:
    ; Add module to array
    mov rax, [r12 + SEQUENTIAL_SIZE]
    mov rcx, [r12 + SEQUENTIAL_MODULES]
    mov [rcx + rax*8], r13
    inc rax
    mov [r12 + SEQUENTIAL_SIZE], rax
    
    xor eax, eax                    ; success
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error_memory:
    mov eax, 2                      ; NEURAL_ERR_OUT_OF_MEMORY
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error_null:
    mov eax, 1                      ; NEURAL_ERR_NULL_POINTER
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_sequential_forward - Forward pass through all modules
; Arguments:
;   RDI = NeuralSequential* seq
;   RSI = const NeuralTensor* input
;   RDX = NeuralTensor* output
; Returns:
;   EAX = 0 on success, error code on failure
; =============================================================================
neural_sequential_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32                     ; space for intermediate storage
    
    test rdi, rdi
    jz .error_null
    test rsi, rsi
    jz .error_null
    test rdx, rdx
    jz .error_null
    
    mov r12, rdi                    ; r12 = sequential
    mov r13, rsi                    ; r13 = input
    mov r14, rdx                    ; r14 = final output
    
    ; Check if empty
    mov rax, [r12 + SEQUENTIAL_SIZE]
    test rax, rax
    jz .error_empty
    
    ; Check if we need to allocate intermediates array
    mov al, [r12 + SEQUENTIAL_SAVE_INTER]
    test al, al
    jz .skip_inter_alloc
    
    ; Ensure intermediates array is large enough
    mov rax, [r12 + SEQUENTIAL_SIZE]
    cmp rax, [r12 + SEQUENTIAL_INTER_CAP]
    jle .skip_inter_alloc
    
    ; Allocate/resize intermediates array
    push r12
    push r13
    push r14
    mov rdi, rax
    shl rdi, 3                      ; 8 bytes per pointer
    call mem_alloc
    pop r14
    pop r13
    pop r12
    test rax, rax
    jz .error_memory
    
    ; Free old array if exists
    mov rdi, [r12 + SEQUENTIAL_INTERMEDIATES]
    test rdi, rdi
    jz .store_new_inter
    push rax
    call mem_free
    pop rax
    
.store_new_inter:
    mov [r12 + SEQUENTIAL_INTERMEDIATES], rax
    mov rcx, [r12 + SEQUENTIAL_SIZE]
    mov [r12 + SEQUENTIAL_INTER_CAP], rcx
    
.skip_inter_alloc:
    ; For single module, forward directly to output
    mov rax, [r12 + SEQUENTIAL_SIZE]
    cmp rax, 1
    je .single_module
    
    ; Multiple modules - need intermediate tensors
    ; For now, we'll assume caller provides properly sized intermediate tensors
    ; TODO: Implement automatic intermediate tensor management
    
    xor rbx, rbx                    ; rbx = current module index
    mov r15, r13                    ; r15 = current input
    
.forward_loop:
    cmp rbx, [r12 + SEQUENTIAL_SIZE]
    jge .done
    
    ; Get current module
    mov rcx, [r12 + SEQUENTIAL_MODULES]
    mov rdi, [rcx + rbx*8]          ; module
    mov [rsp], rdi                  ; save module pointer
    
    ; Determine output tensor
    mov rax, [r12 + SEQUENTIAL_SIZE]
    dec rax
    cmp rbx, rax
    je .use_final_output
    
    ; Use intermediate tensor (for now, assume caller handles this)
    ; This is a limitation - proper implementation would need intermediate tensors
    mov rdi, [rsp]                  ; restore module pointer
    mov rsi, r15                    ; input
    mov rdx, r14                    ; output (temporary)
    call [rdi + MODULE_FORWARD_FN]
    test eax, eax
    jnz .error
    
    ; Save intermediate if enabled
    mov al, [r12 + SEQUENTIAL_SAVE_INTER]
    test al, al
    jz .no_save_inter
    mov rcx, [r12 + SEQUENTIAL_INTERMEDIATES]
    test rcx, rcx
    jz .no_save_inter
    mov [rcx + rbx*8], r14          ; save output node/tensor pointer
    
.no_save_inter:
    mov r15, r14                    ; next input is current output
    jmp .next
    
.use_final_output:
    mov rdi, [rsp]                  ; restore module pointer
    mov rsi, r15                    ; input
    mov rdx, r14                    ; final output
    call [rdi + MODULE_FORWARD_FN]
    test eax, eax
    jnz .error
    
    ; Save final output as intermediate too if enabled
    mov al, [r12 + SEQUENTIAL_SAVE_INTER]
    test al, al
    jz .next
    mov rcx, [r12 + SEQUENTIAL_INTERMEDIATES]
    test rcx, rcx
    jz .next
    mov [rcx + rbx*8], r14
    
.next:
    inc rbx
    jmp .forward_loop
    
.single_module:
    mov rcx, [r12 + SEQUENTIAL_MODULES]
    mov rdi, [rcx]                  ; first (only) module
    mov rsi, r13                    ; input
    mov rdx, r14                    ; output
    call [rdi + MODULE_FORWARD_FN]
    test eax, eax
    jnz .error
    
    ; Save intermediate if enabled
    mov al, [r12 + SEQUENTIAL_SAVE_INTER]
    test al, al
    jz .done
    mov rcx, [r12 + SEQUENTIAL_INTERMEDIATES]
    test rcx, rcx
    jz .done
    mov qword [rcx], r14            ; save single output
    
.done:
    xor eax, eax
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error_memory:
    mov eax, 2                      ; NEURAL_ERR_OUT_OF_MEMORY
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error_empty:
    mov eax, 3                      ; NEURAL_ERR_INVALID_ARGUMENT
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error_null:
    mov eax, 1                      ; NEURAL_ERR_NULL_POINTER
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error:
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_sequential_size - Get number of modules
; Arguments:
;   RDI = NeuralSequential* seq
; Returns:
;   RAX = number of modules
; =============================================================================
neural_sequential_size:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    mov rax, [rdi + SEQUENTIAL_SIZE]
    pop rbp
    ret
    
.null:
    xor eax, eax
    pop rbp
    ret

; =============================================================================
; neural_sequential_get - Get module at index
; Arguments:
;   RDI = NeuralSequential* seq
;   RSI = uint64_t index
; Returns:
;   RAX = NeuralLinear* or NULL
; =============================================================================
neural_sequential_get:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null
    
    cmp rsi, [rdi + SEQUENTIAL_SIZE]
    jge .null
    
    mov rcx, [rdi + SEQUENTIAL_MODULES]
    mov rax, [rcx + rsi*8]
    
    pop rbp
    ret
    
.null:
    xor eax, eax
    pop rbp
    ret

; =============================================================================
; neural_sequential_parameters - Get all parameters
; Arguments:
;   RDI = NeuralSequential* seq
;   RSI = NeuralTensor** params
;   RDX = uint64_t max_params
; Returns:
;   RAX = number of parameters found, or -1 on error
; =============================================================================
neural_sequential_parameters:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    test rdi, rdi
    jz .error
    test rsi, rsi
    jz .error
    
    mov r12, rdi                    ; r12 = sequential
    mov r13, rsi                    ; r13 = params array
    mov r14, rdx                    ; r14 = max_params
    
    xor r15, r15                    ; r15 = param count
    
    ; Iterate through all modules
    xor rbx, rbx                    ; rbx = module index
.param_loop:
    cmp rbx, [r12 + SEQUENTIAL_SIZE]
    jge .done
    
    mov rcx, [r12 + SEQUENTIAL_MODULES]
    mov rdi, [rcx + rbx*8]          ; current module
    test rdi, rdi
    jz .next_module                 ; skip null modules
    
    ; Get params array from module
    mov rax, [rdi + MODULE_PARAMS]  ; get params array
    test rax, rax
    jz .next_module
    
    ; Get weight tensor (params[0])
    mov rsi, [rax]                  ; weight tensor
    test rsi, rsi
    jz .check_bias
    cmp r15, r14
    jge .done                       ; no more room
    mov [r13 + r15*8], rsi
    inc r15
    
.check_bias:
    ; Get bias tensor (params[1])
    mov rax, [rdi + MODULE_PARAMS]  ; get params array again
    test rax, rax
    jz .next_module
    mov rsi, [rax + 8]              ; bias tensor
    test rsi, rsi
    jz .next_module
    cmp r15, r14
    jge .done                       ; no more room
    mov [r13 + r15*8], rsi
    inc r15
    
.next_module:
    inc rbx
    jmp .param_loop
    
.done:
    mov rax, r15
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.error:
    mov rax, -1
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; neural_sequential_get_intermediate - Get intermediate output at index
; Arguments:
;   RDI = NeuralSequential* seq
;   RSI = uint64_t index
; Returns:
;   RAX = intermediate output pointer (Node* or Tensor*) or NULL
; =============================================================================
neural_sequential_get_intermediate:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .null_inter
    
    ; Check if intermediates are enabled
    mov al, [rdi + SEQUENTIAL_SAVE_INTER]
    test al, al
    jz .null_inter
    
    ; Check bounds
    cmp rsi, [rdi + SEQUENTIAL_SIZE]
    jge .null_inter
    
    ; Check if array exists
    mov rcx, [rdi + SEQUENTIAL_INTERMEDIATES]
    test rcx, rcx
    jz .null_inter
    
    ; Return intermediate at index
    mov rax, [rcx + rsi*8]
    pop rbp
    ret
    
.null_inter:
    xor eax, eax
    pop rbp
    ret

; =============================================================================
; neural_sequential_set_save_intermediates - Enable/disable saving intermediates
; Arguments:
;   RDI = NeuralSequential* seq
;   RSI = uint8_t enable (0 = disable, non-zero = enable)
; Returns:
;   nothing
; =============================================================================
neural_sequential_set_save_intermediates:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .done_save_inter
    
    mov [rdi + SEQUENTIAL_SAVE_INTER], sil
    
.done_save_inter:
    pop rbp
    ret

; =============================================================================
; neural_sequential_clear_intermediates - Clear saved intermediate outputs
; Arguments:
;   RDI = NeuralSequential* seq
; Returns:
;   nothing
; =============================================================================
neural_sequential_clear_intermediates:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    test rdi, rdi
    jz .done_clear_inter
    
    mov r12, rdi
    
    ; Get intermediates array
    mov rbx, [r12 + SEQUENTIAL_INTERMEDIATES]
    test rbx, rbx
    jz .done_clear_inter
    
    ; Zero out all entries
    xor rcx, rcx
.clear_loop:
    cmp rcx, [r12 + SEQUENTIAL_SIZE]
    jge .done_clear_inter
    mov qword [rbx + rcx*8], 0
    inc rcx
    jmp .clear_loop
    
.done_clear_inter:
    pop r12
    pop rbx
    pop rbp
    ret

; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
