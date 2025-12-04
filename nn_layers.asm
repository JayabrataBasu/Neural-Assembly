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

%define NODE_VALUE          0
%define NODE_GRAD           8
%define NODE_BACKWARD_FN    16
%define NODE_N_PARENTS      24
%define NODE_VISITED        28
%define NODE_PARENTS        32
%define NODE_SAVED_TENSORS  40

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

; Export layer functions
global linear_create
global linear_forward
global linear_forward_fn
global linear_backward
global conv2d_create
global conv2d_forward
global conv2d_forward_fn
global conv2d_backward
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
    movsd [rsp], xmm1               ; save scale
    
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
    mulsd xmm0, [rsp+8]             ; Note: scale is at [rsp] but we pushed rcx
    
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
    mov [rsp], r13                  ; shape[0] = out_features
    mov [rsp+8], r12                ; shape[1] = in_features
    mov rdi, 2
    lea rsi, [rsp]
    mov edx, r14d
    call tensor_create
    mov [rbx], rax                  ; params[0] = weight
    mov [rsp+16], rax
    
    ; Initialize weight with Xavier
    mov rdi, rax
    mov rsi, r12                    ; fan_in
    mov rdx, r13                    ; fan_out
    call xavier_init
    
    ; Create bias tensor (out_features)
    mov [rsp], r13                  ; shape[0] = out_features
    mov rdi, 1
    lea rsi, [rsp]
    mov edx, r14d
    call tensor_create
    mov [rbx + 8], rax              ; params[1] = bias
    ; Bias initialized to zeros (default)
    
    ; Allocate param_nodes array
    mov rdi, 16
    call mem_alloc
    mov [r15 + MODULE_PARAM_NODES], rax
    mov [rsp+24], rax
    
    ; Create nodes for parameters
    mov rdi, [rbx]                  ; weight tensor
    mov rsi, 1                      ; requires_grad
    call node_create
    mov rcx, [rsp+24]
    mov [rcx], rax                  ; param_nodes[0]
    
    mov rdi, [rbx + 8]              ; bias tensor
    mov rsi, 1
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
    mov rcx, [rax]                  ; weight_node
    mov [rsp], rcx
    mov rcx, [rax + 8]              ; bias_node
    mov [rsp+8], rcx
    
    ; Forward: y = x @ W^T + b
    ; First create W^T
    mov rax, [r12 + MODULE_PARAMS]
    mov rdi, [rax]                  ; weight tensor (out x in)
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
    mov rcx, [rax]                  ; batch_size
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
    mov [rax], r13                  ; parent[0] = input
    mov rdx, [rsp]
    mov [rax + 8], rdx              ; parent[1] = weight
    mov rdx, [rsp+8]
    mov [rax + 16], rdx             ; parent[2] = bias
    
    ; Save module reference for backward
    mov [rcx + NODE_SAVED_TENSORS], r12
    
    ; Store output
    mov [r14], rcx
    
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
    mov rax, [r14]                  ; input node
    mov [rsp], rax
    mov rax, [r14 + 8]              ; weight node
    mov [rsp+8], rax
    mov rax, [r14 + 16]             ; bias node
    mov [rsp+16], rax
    
    ; Get dimensions from dL/dout
    mov rax, [r13 + TENSOR_SHAPE]
    mov rcx, [rax]                  ; batch_size
    mov [rsp+24], rcx
    mov rcx, [rax + 8]              ; out_features
    mov [rsp+32], rcx
    
    ; Get input shape
    mov rax, [rsp]
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
    mov rax, [rsp]                  ; input node
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
    mov rax, [rsp]                  ; input node
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
; conv2d_create - Create a Conv2d layer (placeholder - basic implementation)
; Arguments:
;   RDI = in_channels
;   RSI = out_channels
;   RDX = kernel_h
;   RCX = kernel_w
;   R8 = stride
;   R9 = padding
;   [stack] = dtype
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
    mov [rsp], rdi                  ; in_channels
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
    mov rax, [rsp]
    mov [rbx], rax                  ; in_channels
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
    mov rax, [rsp]                  ; in_channels
    mov [rsp+60], rax
    ; Note: For simplicity, treating as 2D (out x in*kh*kw)
    mov rax, [rsp]
    imul rax, [rsp+16]
    imul rax, [rsp+24]              ; in * kh * kw
    mov [rsp+60], rax
    
    mov rdi, 2
    lea rsi, [rsp+52]
    mov edx, [rsp+48]
    call tensor_create
    mov [r12], rax
    
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
    
    mov rdi, [r12]
    mov rsi, 1
    call node_create
    mov [r13], rax
    
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
    mov [rdx], rax
    
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
    mov [rdx], eax
    
    mov rax, [rdi + MODULE_PARAMS]
    mov [rsi], rax
    
    pop rbp
    ret

.null_module:
    mov dword [rdx], 0
    mov qword [rsi], 0
    pop rbp
    ret
