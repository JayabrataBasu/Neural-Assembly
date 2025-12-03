; =============================================================================
; losses.asm - Loss Functions with Autograd Support
; =============================================================================
; MSE Loss, Cross-Entropy Loss
; =============================================================================

%define NODE_VALUE          0
%define NODE_GRAD           8
%define NODE_BACKWARD_FN    16
%define NODE_N_PARENTS      24
%define NODE_VISITED        28
%define NODE_PARENTS        32
%define NODE_SAVED_TENSORS  40
%define NODE_N_SAVED        48

%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_DTYPE        32

%define DT_FLOAT32          0
%define DT_FLOAT64          1

section .data
    align 8
    two_f64:            dq 2.0
    one_f64:            dq 1.0
    neg_one_f64:        dq -1.0
    eps_f64:            dq 1.0e-7

section .bss
    align 32

section .text

; External functions
extern mem_alloc
extern mem_free
extern tensor_create
extern tensor_zeros
extern tensor_free
extern tensor_numel
extern tensor_fill
extern node_create
extern ew_add
extern ew_sub
extern ew_mul
extern ew_scalar_mul
extern reduce_sum
extern exp
extern log

; Export loss functions
global mse_loss
global mse_loss_backward
global cross_entropy_loss
global cross_entropy_loss_backward

; =============================================================================
; mse_loss - Mean Squared Error loss
; Arguments:
;   RDI = Node* pred (predictions)
;   RSI = Node* target (ground truth)
; Returns:
;   RAX = Node* loss (scalar)
; =============================================================================
mse_loss:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; pred node
    mov r13, rsi                    ; target node (not a Node, just Tensor*)
    
    ; Get prediction tensor
    mov rax, [r12 + NODE_VALUE]
    mov [rsp], rax                  ; pred tensor
    mov r14, rax
    
    ; Get element count
    mov rdi, r14
    call tensor_numel
    mov r15, rax                    ; numel
    
    ; Create diff tensor (pred - target)
    mov rdi, [r14 + TENSOR_NDIM]
    mov rsi, [r14 + TENSOR_SHAPE]
    mov edx, [r14 + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+8], rax                ; diff tensor
    mov rbx, rax
    
    ; Compute diff = pred - target
    ; Note: target is passed as Node*, get its value
    mov rdi, rbx
    mov rsi, r14
    mov rax, [r13 + NODE_VALUE]
    mov rdx, rax
    call ew_sub
    
    ; Square the diff: diff = diff * diff
    mov rdi, rbx
    mov rsi, rbx
    mov rdx, rbx
    call ew_mul
    
    ; Create scalar output tensor for sum
    mov qword [rsp+16], 1
    mov rdi, 1
    lea rsi, [rsp+16]
    mov edx, [rbx + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+24], rax               ; sum tensor
    
    ; Sum all squared differences
    mov rdi, rbx
    mov rsi, -1                     ; sum all
    mov rdx, rax
    call reduce_sum
    
    ; Divide by numel to get mean
    mov rdi, [rsp+24]
    mov rdi, [rdi + TENSOR_DATA]
    mov eax, [rbx + TENSOR_DTYPE]
    
    cmp eax, DT_FLOAT32
    je .mean_f32
    
    ; float64
    movsd xmm0, [rdi]
    cvtsi2sd xmm1, r15
    divsd xmm0, xmm1
    movsd [rdi], xmm0
    jmp .create_loss_node

.mean_f32:
    movss xmm0, [rdi]
    cvtsi2ss xmm1, r15
    divss xmm0, xmm1
    movss [rdi], xmm0

.create_loss_node:
    mov rdi, [rsp+24]
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov [rsp+32], rax               ; loss node
    
    ; Set backward function
    lea rcx, [rel mse_loss_backward]
    mov [rax + NODE_BACKWARD_FN], rcx
    
    ; Set parents (pred, target)
    mov dword [rax + NODE_N_PARENTS], 2
    push rax
    mov rdi, 16
    call mem_alloc
    pop rcx
    mov [rcx + NODE_PARENTS], rax
    mov [rax], r12                  ; parent[0] = pred
    mov [rax + 8], r13              ; parent[1] = target
    
    ; Save diff tensor for backward
    push rcx
    mov rdi, 8
    call mem_alloc
    pop rcx
    mov [rcx + NODE_SAVED_TENSORS], rax
    mov rdx, [rsp+8]                ; diff tensor
    mov [rax], rdx
    mov qword [rcx + NODE_N_SAVED], 1
    
    mov rax, [rsp+32]
    
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    ; Cleanup
    mov rdi, [rsp+8]
    call tensor_free
    mov rdi, [rsp+24]
    call tensor_free
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
; mse_loss_backward - Backward for MSE loss
; dL/dpred = (2/n) * (pred - target) * dL/dloss
; =============================================================================
mse_loss_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self node
    mov r13, [r12 + NODE_GRAD]      ; dL/dloss (scalar)
    mov r14, [r12 + NODE_PARENTS]
    
    ; Get pred node and its grad
    mov rax, [r14]                  ; pred node
    mov r15, [rax + NODE_GRAD]
    test r15, r15
    jz .done
    
    ; Get saved diff tensor (pred - target)
    mov rax, [r12 + NODE_SAVED_TENSORS]
    mov rbx, [rax]                  ; diff tensor
    
    ; Get numel
    mov rdi, rbx
    call tensor_numel
    mov [rsp], rax                  ; numel
    
    ; Get dL/dloss value
    mov rdi, [r13 + TENSOR_DATA]
    mov eax, [r13 + TENSOR_DTYPE]
    
    cmp eax, DT_FLOAT32
    je .bwd_f32
    
    ; float64
    movsd xmm0, [rdi]               ; dL/dloss
    
    ; Compute scale = 2 * dL/dloss / numel
    mulsd xmm0, [rel two_f64]
    cvtsi2sd xmm1, qword [rsp]
    divsd xmm0, xmm1                ; scale = 2 * dL/dloss / n
    
    ; grad_pred += scale * diff
    mov rdi, [r15 + TENSOR_DATA]
    mov rsi, [rbx + TENSOR_DATA]
    mov rcx, [rsp]
    
    xor r8, r8
.bwd_f64_loop:
    cmp r8, rcx
    jge .done
    
    movsd xmm1, [rsi + r8*8]        ; diff[i]
    mulsd xmm1, xmm0                ; scale * diff[i]
    addsd xmm1, [rdi + r8*8]
    movsd [rdi + r8*8], xmm1
    
    inc r8
    jmp .bwd_f64_loop

.bwd_f32:
    movss xmm0, [rdi]
    cvtss2sd xmm0, xmm0
    mulsd xmm0, [rel two_f64]
    cvtsi2sd xmm1, qword [rsp]
    divsd xmm0, xmm1
    cvtsd2ss xmm0, xmm0
    
    mov rdi, [r15 + TENSOR_DATA]
    mov rsi, [rbx + TENSOR_DATA]
    mov rcx, [rsp]
    
    xor r8, r8
.bwd_f32_loop:
    cmp r8, rcx
    jge .done
    
    movss xmm1, [rsi + r8*4]
    mulss xmm1, xmm0
    addss xmm1, [rdi + r8*4]
    movss [rdi + r8*4], xmm1
    
    inc r8
    jmp .bwd_f32_loop

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
; cross_entropy_loss - Cross-entropy loss with softmax
; Arguments:
;   RDI = Node* logits (batch x num_classes)
;   RSI = Tensor* target_indices (batch,) - integer class indices
; Returns:
;   RAX = Node* loss (scalar)
; =============================================================================
cross_entropy_loss:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 88
    
    mov r12, rdi                    ; logits node
    mov r13, rsi                    ; target indices tensor
    
    ; Get logits tensor
    mov rax, [r12 + NODE_VALUE]
    mov [rsp], rax                  ; logits tensor
    mov r14, rax
    
    ; Get shape
    mov rax, [r14 + TENSOR_SHAPE]
    mov rcx, [rax]                  ; batch_size
    mov [rsp+8], rcx
    mov rcx, [rax + 8]              ; num_classes
    mov [rsp+16], rcx
    
    ; Create softmax output tensor
    mov rdi, [r14 + TENSOR_NDIM]
    mov rsi, [r14 + TENSOR_SHAPE]
    mov edx, [r14 + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+24], rax               ; softmax tensor
    mov r15, rax
    
    ; Compute softmax (inline for efficiency)
    mov rdi, [r15 + TENSOR_DATA]
    mov rsi, [r14 + TENSOR_DATA]
    mov [rsp+32], rdi               ; softmax data
    mov [rsp+40], rsi               ; logits data
    
    mov eax, [r15 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .softmax_f32
    
    ; float64 softmax
    xor rbx, rbx                    ; batch index
.softmax_f64_batch:
    cmp rbx, [rsp+8]
    jge .compute_loss
    
    ; Find max for numerical stability
    mov rcx, rbx
    imul rcx, [rsp+16]
    shl rcx, 3
    mov rsi, [rsp+40]
    add rsi, rcx
    mov rdi, [rsp+32]
    add rdi, rcx
    
    mov r8, [rsp+16]                ; num_classes
    movsd xmm0, [rsi]               ; max = logits[0]
    mov r9, 1
.find_max_f64:
    cmp r9, r8
    jge .exp_sum_f64
    maxsd xmm0, [rsi + r9*8]
    inc r9
    jmp .find_max_f64

.exp_sum_f64:
    movsd [rsp+48], xmm0            ; save max
    vxorpd xmm2, xmm2, xmm2         ; sum = 0
    
    xor r9, r9
.exp_loop_f64:
    cmp r9, r8
    jge .normalize_f64
    
    push rdi
    push rsi
    push r8
    push r9
    push rbx
    
    movsd xmm0, [rsi + r9*8]
    subsd xmm0, [rsp+88]            ; x - max (adjusted for pushes)
    sub rsp, 8
    call exp wrt ..plt
    add rsp, 8
    
    pop rbx
    pop r9
    pop r8
    pop rsi
    pop rdi
    
    movsd [rdi + r9*8], xmm0
    addsd xmm2, xmm0
    movsd [rsp+56], xmm2
    
    inc r9
    jmp .exp_loop_f64

.normalize_f64:
    movsd xmm2, [rsp+56]
    xor r9, r9
.norm_loop_f64:
    cmp r9, r8
    jge .next_batch_f64
    movsd xmm0, [rdi + r9*8]
    divsd xmm0, xmm2
    movsd [rdi + r9*8], xmm0
    inc r9
    jmp .norm_loop_f64

.next_batch_f64:
    inc rbx
    jmp .softmax_f64_batch

.softmax_f32:
    ; Similar for float32
    xor rbx, rbx
.softmax_f32_batch:
    cmp rbx, [rsp+8]
    jge .compute_loss
    
    mov rcx, rbx
    imul rcx, [rsp+16]
    shl rcx, 2
    mov rsi, [rsp+40]
    add rsi, rcx
    mov rdi, [rsp+32]
    add rdi, rcx
    
    mov r8, [rsp+16]
    movss xmm0, [rsi]
    mov r9, 1
.find_max_f32:
    cmp r9, r8
    jge .exp_sum_f32
    maxss xmm0, [rsi + r9*4]
    inc r9
    jmp .find_max_f32

.exp_sum_f32:
    movss [rsp+48], xmm0
    vxorps xmm2, xmm2, xmm2
    
    xor r9, r9
.exp_loop_f32:
    cmp r9, r8
    jge .normalize_f32
    
    push rdi
    push rsi
    push r8
    push r9
    push rbx
    
    movss xmm0, [rsi + r9*4]
    cvtss2sd xmm0, xmm0
    movss xmm1, [rsp+88]
    cvtss2sd xmm1, xmm1
    subsd xmm0, xmm1
    sub rsp, 8
    call exp wrt ..plt
    add rsp, 8
    cvtsd2ss xmm0, xmm0
    
    pop rbx
    pop r9
    pop r8
    pop rsi
    pop rdi
    
    movss [rdi + r9*4], xmm0
    addss xmm2, xmm0
    movss [rsp+56], xmm2
    
    inc r9
    jmp .exp_loop_f32

.normalize_f32:
    movss xmm2, [rsp+56]
    xor r9, r9
.norm_loop_f32:
    cmp r9, r8
    jge .next_batch_f32
    movss xmm0, [rdi + r9*4]
    divss xmm0, xmm2
    movss [rdi + r9*4], xmm0
    inc r9
    jmp .norm_loop_f32

.next_batch_f32:
    inc rbx
    jmp .softmax_f32_batch

.compute_loss:
    ; Compute -log(softmax[target]) for each sample and average
    mov qword [rsp+48], 1
    mov rdi, 1
    lea rsi, [rsp+48]
    mov edx, [r15 + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+64], rax               ; loss tensor
    
    vxorpd xmm0, xmm0, xmm0         ; sum = 0
    
    mov rdi, [rsp+32]               ; softmax data
    mov rsi, [r13 + TENSOR_DATA]    ; target indices
    mov rcx, [rsp+8]                ; batch_size
    mov r8, [rsp+16]                ; num_classes
    
    mov eax, [r15 + TENSOR_DTYPE]
    cmp eax, DT_FLOAT32
    je .loss_f32
    
    ; float64
    xor rbx, rbx
.loss_f64_loop:
    cmp rbx, rcx
    jge .store_loss_f64
    
    ; Get target class for this sample
    mov r9, [rsi + rbx*8]           ; target_indices[b] (assuming uint64)
    
    ; Get softmax[b, target]
    mov rax, rbx
    imul rax, r8
    add rax, r9
    
    push rdi
    push rsi
    push rcx
    push rbx
    movsd xmm0, [rdi + rax*8]
    addsd xmm0, [rel eps_f64]       ; add epsilon for stability
    sub rsp, 8
    call log wrt ..plt
    add rsp, 8
    pop rbx
    pop rcx
    pop rsi
    pop rdi
    
    ; Negate and add to sum
    movsd xmm1, [rel neg_one_f64]
    mulsd xmm0, xmm1
    movsd xmm1, [rsp+72]
    addsd xmm1, xmm0
    movsd [rsp+72], xmm1
    
    inc rbx
    jmp .loss_f64_loop

.store_loss_f64:
    ; Average
    movsd xmm0, [rsp+72]
    cvtsi2sd xmm1, qword [rsp+8]
    divsd xmm0, xmm1
    
    mov rdi, [rsp+64]
    mov rdi, [rdi + TENSOR_DATA]
    movsd [rdi], xmm0
    jmp .create_node

.loss_f32:
    xor rbx, rbx
.loss_f32_loop:
    cmp rbx, rcx
    jge .store_loss_f32
    
    mov r9d, [rsi + rbx*4]          ; target as int32
    
    mov rax, rbx
    imul rax, r8
    add rax, r9
    
    push rdi
    push rsi
    push rcx
    push rbx
    movss xmm0, [rdi + rax*4]
    cvtss2sd xmm0, xmm0
    addsd xmm0, [rel eps_f64]
    sub rsp, 8
    call log wrt ..plt
    add rsp, 8
    pop rbx
    pop rcx
    pop rsi
    pop rdi
    
    movsd xmm1, [rel neg_one_f64]
    mulsd xmm0, xmm1
    cvtsd2ss xmm0, xmm0
    movss xmm1, [rsp+72]
    addss xmm1, xmm0
    movss [rsp+72], xmm1
    
    inc rbx
    jmp .loss_f32_loop

.store_loss_f32:
    movss xmm0, [rsp+72]
    cvtsi2ss xmm1, dword [rsp+8]
    divss xmm0, xmm1
    
    mov rdi, [rsp+64]
    mov rdi, [rdi + TENSOR_DATA]
    movss [rdi], xmm0

.create_node:
    mov rdi, [rsp+64]
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov [rsp+80], rax
    
    ; Set backward function
    lea rcx, [rel cross_entropy_loss_backward]
    mov [rax + NODE_BACKWARD_FN], rcx
    
    ; Set parent (logits)
    mov dword [rax + NODE_N_PARENTS], 1
    push rax
    mov rdi, 8
    call mem_alloc
    pop rcx
    mov [rcx + NODE_PARENTS], rax
    mov [rax], r12
    
    ; Save softmax and target for backward
    push rcx
    mov rdi, 16
    call mem_alloc
    pop rcx
    mov [rcx + NODE_SAVED_TENSORS], rax
    mov rdx, [rsp+24]               ; softmax tensor
    mov [rax], rdx
    mov [rax + 8], r13              ; target indices
    mov qword [rcx + NODE_N_SAVED], 2
    
    mov rax, [rsp+80]
    
    add rsp, 88
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 88
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; cross_entropy_loss_backward - Backward for cross-entropy loss
; dL/dlogits = (softmax - one_hot(target)) / batch_size
; =============================================================================
cross_entropy_loss_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self node
    mov r13, [r12 + NODE_GRAD]      ; dL/dloss (scalar)
    
    ; Get parent (logits) grad
    mov rax, [r12 + NODE_PARENTS]
    mov rax, [rax]                  ; logits node
    mov r14, [rax + NODE_GRAD]
    test r14, r14
    jz .done
    
    ; Get saved tensors
    mov rax, [r12 + NODE_SAVED_TENSORS]
    mov r15, [rax]                  ; softmax tensor
    mov rbx, [rax + 8]              ; target indices tensor
    
    ; Get shape
    mov rax, [r15 + TENSOR_SHAPE]
    mov rcx, [rax]                  ; batch_size
    mov [rsp], rcx
    mov rcx, [rax + 8]              ; num_classes
    mov [rsp+8], rcx
    
    ; Get dL/dloss value
    mov rdi, [r13 + TENSOR_DATA]
    mov eax, [r13 + TENSOR_DTYPE]
    
    cmp eax, DT_FLOAT32
    je .bwd_f32
    
    ; float64: grad = (softmax - one_hot) * dL/dloss / batch_size
    movsd xmm2, [rdi]               ; dL/dloss
    cvtsi2sd xmm3, qword [rsp]      ; batch_size
    divsd xmm2, xmm3                ; scale = dL/dloss / batch
    
    mov rdi, [r14 + TENSOR_DATA]    ; grad data
    mov rsi, [r15 + TENSOR_DATA]    ; softmax data
    mov rdx, [rbx + TENSOR_DATA]    ; target indices
    
    mov rcx, [rsp]                  ; batch_size
    mov r8, [rsp+8]                 ; num_classes
    
    xor r9, r9                      ; batch idx
.bwd_f64_batch:
    cmp r9, rcx
    jge .done
    
    mov r10, [rdx + r9*8]           ; target class
    
    xor r11, r11                    ; class idx
.bwd_f64_class:
    cmp r11, r8
    jge .bwd_f64_next_batch
    
    mov rax, r9
    imul rax, r8
    add rax, r11
    
    movsd xmm0, [rsi + rax*8]       ; softmax[b,c]
    
    ; Subtract 1 if this is the target class
    cmp r11, r10
    jne .not_target_f64
    subsd xmm0, [rel one_f64]
.not_target_f64:
    
    mulsd xmm0, xmm2                ; * scale
    addsd xmm0, [rdi + rax*8]       ; accumulate
    movsd [rdi + rax*8], xmm0
    
    inc r11
    jmp .bwd_f64_class

.bwd_f64_next_batch:
    inc r9
    jmp .bwd_f64_batch

.bwd_f32:
    movss xmm2, [rdi]
    cvtss2sd xmm2, xmm2
    cvtsi2sd xmm3, qword [rsp]
    divsd xmm2, xmm3
    cvtsd2ss xmm2, xmm2
    
    mov rdi, [r14 + TENSOR_DATA]
    mov rsi, [r15 + TENSOR_DATA]
    mov rdx, [rbx + TENSOR_DATA]
    
    mov rcx, [rsp]
    mov r8, [rsp+8]
    
    xor r9, r9
.bwd_f32_batch:
    cmp r9, rcx
    jge .done
    
    mov r10d, [rdx + r9*4]
    
    xor r11, r11
.bwd_f32_class:
    cmp r11, r8
    jge .bwd_f32_next_batch
    
    mov rax, r9
    imul rax, r8
    add rax, r11
    
    movss xmm0, [rsi + rax*4]
    
    cmp r11, r10
    jne .not_target_f32
    mov eax, 0x3f800000             ; 1.0f
    movd xmm1, eax
    subss xmm0, xmm1
.not_target_f32:
    
    mov rax, r9
    imul rax, r8
    add rax, r11
    
    mulss xmm0, xmm2
    addss xmm0, [rdi + rax*4]
    movss [rdi + rax*4], xmm0
    
    inc r11
    jmp .bwd_f32_class

.bwd_f32_next_batch:
    inc r9
    jmp .bwd_f32_batch

.done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
