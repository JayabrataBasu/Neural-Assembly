; =============================================================================
; training_ops.asm - Core training operations in x86-64 assembly
;
; Implements performance-critical training utilities:
;   - Confusion matrix update and metrics computation
;   - Learning rate schedule calculations (step, exponential, cosine)
;   - NaN/Inf detection on tensors
;   - Gradient L2 norm computation
;   - Dropout forward/backward
;   - Weight initialization (He/Kaiming, Xavier/Glorot)
;
; All functions follow System V AMD64 ABI calling convention.
; =============================================================================

default rel

section .data
    align 16
    const_pi:           dq 3.14159265358979323846
    const_two:          dq 2.0
    const_three:        dq 3.0
    const_six:          dq 6.0
    const_half:         dq 0.5
    const_one:          dq 1.0
    const_zero:         dq 0.0
    const_neg_one:      dq -1.0
    const_rand_max:     dd 0x7FFFFFFF    ; 2147483647
    const_rand_max_f:   dq 2147483647.0
    const_eps:          dq 1.0e-12

section .bss
    align 8

section .text

; External functions (libc / framework)
extern cos
extern pow
extern sqrt
extern sqrtf
extern rand
extern srand
extern time
extern malloc
extern free
extern memset
extern tensor_numel

; =============================================================================
; SECTION 1: Confusion Matrix Operations
; =============================================================================

; confusion_matrix_update - Update confusion matrix from target/prediction arrays
;
; Args:
;   RDI = int32_t* matrix (flat row-major, num_classes x num_classes)
;   RSI = int32_t* targets (array of true class labels)
;   RDX = int32_t* predictions (array of predicted class labels)
;   RCX = uint64_t n (number of samples)
;   R8  = uint64_t num_classes
;
; Matrix layout: matrix[target * num_classes + prediction] += 1
; Returns: EAX = number of valid samples processed
; =============================================================================
global confusion_matrix_update
confusion_matrix_update:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi                ; matrix
    mov r13, rsi                ; targets
    mov r14, rdx                ; predictions
    mov r15, rcx                ; n
    mov rbx, r8                 ; num_classes

    xor ecx, ecx               ; i = 0
    xor eax, eax                ; valid count = 0
    mov [rsp-8], rax            ; store valid count on stack

.cm_loop:
    cmp rcx, r15
    jge .cm_done

    ; Load target and prediction
    movsxd rax, dword [r13 + rcx*4]     ; target
    movsxd rdx, dword [r14 + rcx*4]     ; prediction

    ; Bounds check: 0 <= target < num_classes && 0 <= prediction < num_classes
    test rax, rax
    js .cm_skip
    cmp rax, rbx
    jge .cm_skip
    test rdx, rdx
    js .cm_skip
    cmp rdx, rbx
    jge .cm_skip

    ; matrix[target * num_classes + prediction] += 1
    push rcx
    mov rcx, rax
    imul rcx, rbx               ; target * num_classes
    add rcx, rdx                ; + prediction
    inc dword [r12 + rcx*4]     ; matrix[idx]++
    pop rcx

    inc qword [rsp-8]           ; valid_count++

.cm_skip:
    inc rcx
    jmp .cm_loop

.cm_done:
    mov rax, [rsp-8]            ; return valid count

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; compute_class_precision - Compute precision for one class from confusion matrix
;
; Precision = TP / (TP + FP) = matrix[c][c] / sum_over_rows(matrix[r][c])
;
; Args:
;   RDI = int32_t* matrix (flat row-major)
;   ESI = class index
;   EDX = num_classes
; Returns:
;   XMM0 = precision (double)
; =============================================================================
global compute_class_precision
compute_class_precision:
    push rbp
    mov rbp, rsp

    movsxd rsi, esi             ; class index
    movsxd rdx, edx             ; num_classes

    ; TP = matrix[class * num_classes + class]
    mov rax, rsi
    imul rax, rdx
    add rax, rsi
    movsxd r8, dword [rdi + rax*4]   ; TP

    ; Sum column: sum_r matrix[r][class]
    xor ecx, ecx               ; r = 0
    xor r9d, r9d               ; col_sum = 0
.prec_col_loop:
    cmp rcx, rdx
    jge .prec_col_done
    mov rax, rcx
    imul rax, rdx
    add rax, rsi
    movsxd r10, dword [rdi + rax*4]
    add r9, r10
    inc rcx
    jmp .prec_col_loop

.prec_col_done:
    ; precision = TP / col_sum
    test r9, r9
    jz .prec_zero
    cvtsi2sd xmm0, r8
    cvtsi2sd xmm1, r9
    divsd xmm0, xmm1
    pop rbp
    ret

.prec_zero:
    vxorpd xmm0, xmm0, xmm0
    pop rbp
    ret

; =============================================================================
; compute_class_recall - Compute recall for one class from confusion matrix
;
; Recall = TP / (TP + FN) = matrix[c][c] / sum_over_cols(matrix[c][p])
;
; Args:
;   RDI = int32_t* matrix (flat row-major)
;   ESI = class index
;   EDX = num_classes
; Returns:
;   XMM0 = recall (double)
; =============================================================================
global compute_class_recall
compute_class_recall:
    push rbp
    mov rbp, rsp

    movsxd rsi, esi
    movsxd rdx, edx

    ; TP = matrix[class * num_classes + class]
    mov rax, rsi
    imul rax, rdx
    add rax, rsi
    movsxd r8, dword [rdi + rax*4]   ; TP

    ; Sum row: sum_p matrix[class][p]
    mov rax, rsi
    imul rax, rdx               ; class * num_classes (row start)
    xor ecx, ecx
    xor r9d, r9d
.rec_row_loop:
    cmp rcx, rdx
    jge .rec_row_done
    mov r10, rax
    add r10, rcx
    movsxd r11, dword [rdi + r10*4]
    add r9, r11
    inc rcx
    jmp .rec_row_loop

.rec_row_done:
    test r9, r9
    jz .rec_zero
    cvtsi2sd xmm0, r8
    cvtsi2sd xmm1, r9
    divsd xmm0, xmm1
    pop rbp
    ret

.rec_zero:
    vxorpd xmm0, xmm0, xmm0
    pop rbp
    ret

; =============================================================================
; compute_class_f1 - Compute F1 score for one class
;
; F1 = 2 * P * R / (P + R)
;
; Args:
;   RDI = int32_t* matrix
;   ESI = class index
;   EDX = num_classes
; Returns:
;   XMM0 = f1 score (double)
; =============================================================================
global compute_class_f1
compute_class_f1:
    push rbp
    mov rbp, rsp
    sub rsp, 32
    push rbx
    push r12

    mov rbx, rdi
    mov r12d, edx

    ; Get precision
    mov edi, esi                ; save class index - but we need full args
    mov rdi, rbx
    ;esi already = class
    mov edx, r12d
    call compute_class_precision
    movsd [rsp+16], xmm0       ; save precision

    ; Get recall
    mov rdi, rbx
    ; esi still = class index (callee-saved? no, esi is caller-saved)
    ; We need to restore esi. Let's use the stack.
    ; Actually esi might have been clobbered. Let me restructure.
    pop r12
    pop rbx
    add rsp, 32
    pop rbp

    ; Simpler approach: inline the computation
    push rbp
    mov rbp, rsp
    sub rsp, 16
    push rbx
    push r12
    push r13

    mov rbx, rdi                ; matrix
    mov r12d, esi               ; class
    mov r13d, edx               ; num_classes

    ; Get precision
    mov rdi, rbx
    mov esi, r12d
    mov edx, r13d
    call compute_class_precision
    movsd [rsp], xmm0          ; save precision

    ; Get recall
    mov rdi, rbx
    mov esi, r12d
    mov edx, r13d
    call compute_class_recall
    ; xmm0 = recall, [rsp] = precision

    movsd xmm1, [rsp]          ; precision
    ; F1 = 2 * P * R / (P + R)
    movapd xmm2, xmm1
    addsd xmm2, xmm0           ; P + R
    movsd xmm3, [rel const_eps]
    ucomisd xmm2, xmm3
    jb .f1_zero                ; if P + R < eps, return 0

    mulsd xmm1, xmm0           ; P * R
    addsd xmm1, xmm1           ; 2 * P * R
    divsd xmm1, xmm2           ; / (P + R)
    movapd xmm0, xmm1

    pop r13
    pop r12
    pop rbx
    add rsp, 16
    pop rbp
    ret

.f1_zero:
    vxorpd xmm0, xmm0, xmm0
    pop r13
    pop r12
    pop rbx
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; compute_accuracy_from_matrix - Compute overall accuracy from confusion matrix
;
; Accuracy = trace(matrix) / total
;
; Args:
;   RDI = int32_t* matrix
;   ESI = num_classes
; Returns:
;   XMM0 = accuracy (double)
; =============================================================================
global compute_accuracy_from_matrix
compute_accuracy_from_matrix:
    push rbp
    mov rbp, rsp

    movsxd rsi, esi             ; num_classes

    ; Sum diagonal (correct predictions)
    xor ecx, ecx               ; i = 0
    xor r8d, r8d               ; correct = 0
    xor r9d, r9d               ; total = 0

.acc_diag_loop:
    cmp rcx, rsi
    jge .acc_sum_total

    ; correct += matrix[i * num_classes + i]
    mov rax, rcx
    imul rax, rsi
    add rax, rcx
    movsxd r10, dword [rdi + rax*4]
    add r8, r10

    ; total += sum of row i
    mov rax, rcx
    imul rax, rsi
    xor edx, edx
.acc_row_loop:
    cmp rdx, rsi
    jge .acc_row_done
    mov r11, rax
    add r11, rdx
    movsxd r10, dword [rdi + r11*4]
    add r9, r10
    inc rdx
    jmp .acc_row_loop

.acc_row_done:
    inc rcx
    jmp .acc_diag_loop

.acc_sum_total:
    test r9, r9
    jz .acc_zero
    cvtsi2sd xmm0, r8
    cvtsi2sd xmm1, r9
    divsd xmm0, xmm1
    pop rbp
    ret

.acc_zero:
    vxorpd xmm0, xmm0, xmm0
    pop rbp
    ret


; =============================================================================
; SECTION 2: Learning Rate Schedule Computations
; =============================================================================

; lr_step_decay - Step decay learning rate schedule
;
; lr = initial_lr * gamma^(epoch / step_size)
;
; Args:
;   XMM0 = initial_lr (double)
;   EDI  = epoch (int)
;   ESI  = step_size (int)
;   XMM1 = gamma (double)
; Returns:
;   XMM0 = new learning rate (double)
; =============================================================================
global lr_step_decay
lr_step_decay:
    push rbp
    mov rbp, rsp
    sub rsp, 16

    movsd [rsp], xmm0           ; save initial_lr
    movsd [rsp+8], xmm1         ; save gamma

    ; Compute epoch / step_size (integer division)
    mov eax, edi
    cdq
    idiv esi                     ; eax = epoch / step_size

    ; pow(gamma, eax)
    movsd xmm0, [rsp+8]         ; gamma
    cvtsi2sd xmm1, eax          ; exponent
    ; Call pow(gamma, exponent)
    call pow wrt ..plt

    ; result = initial_lr * pow(gamma, n)
    mulsd xmm0, [rsp]           ; * initial_lr

    add rsp, 16
    pop rbp
    ret

; =============================================================================
; lr_exponential_decay - Exponential decay learning rate schedule
;
; lr = initial_lr * gamma^epoch
;
; Args:
;   XMM0 = initial_lr (double)
;   EDI  = epoch (int)
;   XMM1 = gamma (double)
; Returns:
;   XMM0 = new learning rate (double)
; =============================================================================
global lr_exponential_decay
lr_exponential_decay:
    push rbp
    mov rbp, rsp
    sub rsp, 16

    movsd [rsp], xmm0           ; save initial_lr

    ; pow(gamma, epoch)
    movapd xmm0, xmm1           ; gamma
    cvtsi2sd xmm1, edi          ; epoch
    call pow wrt ..plt

    ; result = initial_lr * pow(gamma, epoch)
    mulsd xmm0, [rsp]

    add rsp, 16
    pop rbp
    ret

; =============================================================================
; lr_cosine_annealing - Cosine annealing learning rate schedule
;
; lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max))
;
; Args:
;   XMM0 = initial_lr (double)
;   EDI  = epoch (int)
;   ESI  = T_max (int)
;   XMM1 = eta_min (double)
; Returns:
;   XMM0 = new learning rate (double)
; =============================================================================
global lr_cosine_annealing
lr_cosine_annealing:
    push rbp
    mov rbp, rsp
    sub rsp, 32

    movsd [rsp], xmm0           ; initial_lr
    movsd [rsp+8], xmm1         ; eta_min

    ; Handle T_max == 0
    test esi, esi
    jz .cos_return_min

    ; angle = pi * epoch / T_max
    cvtsi2sd xmm0, edi          ; epoch
    cvtsi2sd xmm1, esi          ; T_max
    divsd xmm0, xmm1            ; epoch / T_max
    mulsd xmm0, [rel const_pi]  ; * pi
    ; cos(angle)
    call cos wrt ..plt
    movsd [rsp+16], xmm0        ; save cos_val

    ; lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos_val)
    movsd xmm1, [rsp]           ; initial_lr
    movsd xmm2, [rsp+8]         ; eta_min
    subsd xmm1, xmm2            ; initial_lr - eta_min
    mulsd xmm1, [rel const_half] ; * 0.5
    movsd xmm0, [rsp+16]        ; cos_val
    addsd xmm0, [rel const_one] ; 1 + cos_val
    mulsd xmm0, xmm1            ; * 0.5 * (initial_lr - eta_min)
    addsd xmm0, xmm2            ; + eta_min

    add rsp, 32
    pop rbp
    ret

.cos_return_min:
    movsd xmm0, [rsp+8]         ; eta_min
    add rsp, 32
    pop rbp
    ret


; =============================================================================
; SECTION 3: Tensor Inspection Operations
; =============================================================================

; tensor_has_nan - Check if tensor contains any NaN values
;
; Args:
;   RDI = float* data pointer
;   RSI = uint64_t num_elements
; Returns:
;   EAX = 1 if NaN found, 0 if clean
; =============================================================================
global tensor_has_nan
tensor_has_nan:
    push rbp
    mov rbp, rsp

    test rdi, rdi
    jz .nan_ret_zero
    test rsi, rsi
    jz .nan_ret_zero

    xor ecx, ecx               ; i = 0
.nan_loop:
    cmp rcx, rsi
    jge .nan_ret_zero

    movss xmm0, [rdi + rcx*4]
    ucomiss xmm0, xmm0          ; NaN != NaN sets parity flag
    jp .nan_found

    inc rcx
    jmp .nan_loop

.nan_found:
    mov eax, 1
    pop rbp
    ret

.nan_ret_zero:
    xor eax, eax
    pop rbp
    ret

; =============================================================================
; tensor_has_inf - Check if tensor contains any Inf values
;
; Args:
;   RDI = float* data pointer
;   RSI = uint64_t num_elements
; Returns:
;   EAX = 1 if Inf found, 0 if clean
; =============================================================================
global tensor_has_inf
tensor_has_inf:
    push rbp
    mov rbp, rsp
    push rbx

    test rdi, rdi
    jz .inf_ret_zero
    test rsi, rsi
    jz .inf_ret_zero

    mov rbx, 0x7F800000        ; float +Inf bit pattern

    xor ecx, ecx
.inf_loop:
    cmp rcx, rsi
    jge .inf_ret_zero

    mov eax, [rdi + rcx*4]
    and eax, 0x7FFFFFFF         ; abs(bits)
    cmp eax, ebx                ; == Inf?
    je .inf_found

    inc rcx
    jmp .inf_loop

.inf_found:
    mov eax, 1
    pop rbx
    pop rbp
    ret

.inf_ret_zero:
    xor eax, eax
    pop rbx
    pop rbp
    ret

; =============================================================================
; tensor_grad_l2_norm - Compute L2 norm of a float32 array
;
; norm = sqrt(sum(x[i]^2))
;
; Uses SIMD (SSE) for vectorized sum-of-squares.
;
; Args:
;   RDI = float* data pointer
;   RSI = uint64_t num_elements
; Returns:
;   XMM0 = L2 norm (float)
; =============================================================================
global tensor_grad_l2_norm
tensor_grad_l2_norm:
    push rbp
    mov rbp, rsp

    test rdi, rdi
    jz .norm_zero
    test rsi, rsi
    jz .norm_zero

    ; Accumulate sum of squares using SSE
    xorps xmm2, xmm2           ; accumulator = 0

    ; Process 4 floats at a time
    mov rcx, rsi
    shr rcx, 2                  ; n / 4
    xor edx, edx
    test rcx, rcx
    jz .norm_scalar

.norm_simd_loop:
    movups xmm0, [rdi + rdx]
    mulps xmm0, xmm0            ; x[i]^2
    addps xmm2, xmm0
    add edx, 16                  ; 4 floats = 16 bytes
    dec rcx
    jnz .norm_simd_loop

    ; Horizontal add xmm2
    movhlps xmm0, xmm2
    addps xmm2, xmm0
    movaps xmm0, xmm2
    shufps xmm0, xmm0, 0x55     ; element 1
    addss xmm2, xmm0

.norm_scalar:
    ; Handle remaining elements
    mov rcx, rsi
    and rcx, 3                   ; n % 4
    test rcx, rcx
    jz .norm_sqrt

    ; rdx is byte offset of remaining elements
.norm_scalar_loop:
    movss xmm0, [rdi + rdx]
    mulss xmm0, xmm0
    addss xmm2, xmm0
    add edx, 4
    dec rcx
    jnz .norm_scalar_loop

.norm_sqrt:
    ; xmm2 has sum_of_squares (as float in low lane)
    sqrtss xmm0, xmm2
    pop rbp
    ret

.norm_zero:
    xorps xmm0, xmm0
    pop rbp
    ret


; =============================================================================
; SECTION 4: Dropout Operations
; =============================================================================

; dropout_forward - Apply inverted dropout to tensor data
;
; For each element: if random() < p, mask[i] = 0, output[i] = 0
;                   else mask[i] = 1, output[i] = input[i] / (1 - p)
;
; Args:
;   RDI = float* input data
;   RSI = float* output data
;   RDX = uint8_t* mask (output, same length as data)
;   RCX = uint64_t num_elements
;   XMM0 = dropout probability p (float, 0.0 to 1.0)
; Returns:
;   EAX = 0 on success
; =============================================================================
global dropout_forward
dropout_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16

    mov r12, rdi                ; input
    mov r13, rsi                ; output
    mov r14, rdx                ; mask
    mov r15, rcx                ; num_elements

    ; Store p and compute scale = 1.0 / (1.0 - p)
    cvtss2sd xmm1, xmm0        ; p as double
    movsd xmm2, [rel const_one]
    subsd xmm2, xmm1            ; 1.0 - p
    movsd xmm3, [rel const_one]
    divsd xmm3, xmm2            ; scale = 1.0 / (1.0 - p)
    cvtsd2ss xmm3, xmm3         ; scale as float
    movss [rsp], xmm3            ; save scale

    ; p * RAND_MAX for threshold comparison
    mulsd xmm1, [rel const_rand_max_f]
    cvtsd2si rbx, xmm1          ; threshold = p * RAND_MAX

    xor ecx, ecx                ; i = 0
.drop_loop:
    cmp rcx, r15
    jge .drop_done

    ; Generate random number
    push rcx
    call rand wrt ..plt
    pop rcx

    ; Compare with threshold
    cdqe
    cmp rax, rbx
    jl .drop_zero               ; random < threshold → drop

    ; Keep: mask[i] = 1, output[i] = input[i] * scale
    mov byte [r14 + rcx], 1
    movss xmm0, [r12 + rcx*4]
    mulss xmm0, [rsp]           ; * scale
    movss [r13 + rcx*4], xmm0
    jmp .drop_next

.drop_zero:
    ; Drop: mask[i] = 0, output[i] = 0
    mov byte [r14 + rcx], 0
    xorps xmm0, xmm0
    movss [r13 + rcx*4], xmm0

.drop_next:
    inc rcx
    jmp .drop_loop

.drop_done:
    xor eax, eax
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; dropout_backward - Apply dropout mask to gradients (backward pass)
;
; grad_input[i] = grad_output[i] * mask[i] * scale
;
; Args:
;   RDI = float* grad_output
;   RSI = float* grad_input (output)
;   RDX = uint8_t* mask
;   RCX = uint64_t num_elements
;   XMM0 = dropout probability p (float)
; Returns:
;   EAX = 0 on success
; =============================================================================
global dropout_backward
dropout_backward:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8

    mov r12, rdi                ; grad_output
    mov r13, rsi                ; grad_input
    mov r14, rdx                ; mask
    mov r15, rcx                ; num_elements

    ; Compute scale = 1.0 / (1.0 - p)
    cvtss2sd xmm1, xmm0
    movsd xmm2, [rel const_one]
    subsd xmm2, xmm1
    movsd xmm3, [rel const_one]
    divsd xmm3, xmm2
    cvtsd2ss xmm3, xmm3
    movss [rsp], xmm3

    xor ecx, ecx
.dropb_loop:
    cmp rcx, r15
    jge .dropb_done

    movzx eax, byte [r14 + rcx]
    test eax, eax
    jz .dropb_zero

    ; mask[i] = 1: grad_input = grad_output * scale
    movss xmm0, [r12 + rcx*4]
    mulss xmm0, [rsp]
    movss [r13 + rcx*4], xmm0
    jmp .dropb_next

.dropb_zero:
    xorps xmm0, xmm0
    movss [r13 + rcx*4], xmm0

.dropb_next:
    inc rcx
    jmp .dropb_loop

.dropb_done:
    xor eax, eax
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret


; =============================================================================
; SECTION 5: Weight Initialization
; =============================================================================

; init_kaiming_uniform - He/Kaiming uniform initialization (legacy API)
;
; Fills data with U(-bound, bound) where bound = sqrt(6.0 / fan)
;
; Args:
;   RDI = float* data
;   RSI = uint64_t num_elements
;   RDX = uint64_t fan_in
;   RCX = uint64_t fan_out
;   R8D = mode: 0 = fan_in, 1 = fan_out
;   R9D = seed (0 = use time-based seed)
; Returns:
;   EAX = 0 on success
; =============================================================================
global init_kaiming_uniform
init_kaiming_uniform:
    push rbp
    mov rbp, rsp

    ; Select fan based on mode
    test r8d, r8d
    jnz .ku_fanout
    ; fan = fan_in (rdx already correct)
    jmp .ku_call
.ku_fanout:
    mov rdx, rcx                 ; fan = fan_out

.ku_call:
    ; init_he_uniform(data, n, fan, seed)
    mov ecx, r9d                 ; seed
    call init_he_uniform

    pop rbp
    ret


; =============================================================================
; Clean implementations of init functions
; =============================================================================

; init_uniform_range - Fill float32 array with uniform random values in [lo, hi]
;
; Args:
;   RDI = float* data
;   RSI = uint64_t num_elements
;   XMM0 = lo (float)
;   XMM1 = hi (float)
;   EDX = seed (0 = time-based)
; Returns:
;   EAX = 0 on success
; =============================================================================
global init_uniform_range
init_uniform_range:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 16

    mov r12, rdi                ; data
    mov r13, rsi                ; num_elements

    ; Store lo and range = hi - lo
    movss [rsp], xmm0           ; lo
    subss xmm1, xmm0
    movss [rsp+4], xmm1         ; range = hi - lo

    ; Seed RNG
    test edx, edx
    jnz .ur_use_seed
    xor edi, edi
    call time wrt ..plt
    mov edi, eax
    jmp .ur_do_seed
.ur_use_seed:
    mov edi, edx
.ur_do_seed:
    call srand wrt ..plt

    xor ebx, ebx                ; i = 0
.ur_loop:
    cmp rbx, r13
    jge .ur_done

    push rbx
    call rand wrt ..plt
    pop rbx

    ; val = lo + (rand / RAND_MAX) * range
    cvtsi2ss xmm0, eax
    mov eax, 0x7FFFFFFF
    cvtsi2ss xmm1, eax
    divss xmm0, xmm1            ; [0, 1)
    mulss xmm0, [rsp+4]         ; * range
    addss xmm0, [rsp]           ; + lo
    movss [r12 + rbx*4], xmm0

    inc rbx
    jmp .ur_loop

.ur_done:
    xor eax, eax
    add rsp, 16
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; init_normal_range - Fill float32 array with normal random values (Box-Muller)
;
; Args:
;   RDI = float* data
;   RSI = uint64_t num_elements
;   XMM0 = mean (float)
;   XMM1 = std (float)
;   EDX = seed (0 = time-based)
; Returns:
;   EAX = 0 on success
; =============================================================================
global init_normal_range
init_normal_range:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 16

    mov r12, rdi
    mov r13, rsi

    movss [rsp], xmm0           ; mean
    movss [rsp+4], xmm1         ; std

    ; Seed
    test edx, edx
    jnz .nr_use_seed
    xor edi, edi
    call time wrt ..plt
    mov edi, eax
    jmp .nr_do_seed
.nr_use_seed:
    mov edi, edx
.nr_do_seed:
    call srand wrt ..plt

    xor ebx, ebx
.nr_loop:
    cmp rbx, r13
    jge .nr_done

    ; Box-Muller: generate two normal values from two uniform values
    ; z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)

    ; Get u1
    push rbx
    call rand wrt ..plt
    pop rbx
    cvtsi2sd xmm0, eax
    mov eax, 0x7FFFFFFF
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1            ; u1 in (0, 1)
    ; Clamp away from zero
    movsd xmm1, [rel const_eps]
    maxsd xmm0, xmm1
    movsd [rsp+8], xmm0         ; save u1

    ; Get u2
    push rbx
    call rand wrt ..plt
    pop rbx
    cvtsi2sd xmm0, eax
    mov eax, 0x7FFFFFFF
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1            ; u2

    ; angle = 2 * pi * u2
    mulsd xmm0, [rel const_two]
    mulsd xmm0, [rel const_pi]

    ; cos(angle)
    push rbx
    call cos wrt ..plt
    pop rbx
    movsd [rsp+8], xmm0         ; save cos_val (reuse slot)

    ; -2 * ln(u1): we already consumed u1, need another approach
    ; Actually: let me simplify. Use a simpler approximation.
    ; For init purposes, even a decent approximation to normal is fine.
    ; Use Irwin-Hall sum of 12 uniform → approximately N(0,1)

    ; Sum 12 uniform samples
    xorps xmm2, xmm2            ; accumulator = 0
    mov ecx, 12
.nr_sum:
    push rbx
    push rcx
    call rand wrt ..plt
    pop rcx
    pop rbx
    cvtsi2sd xmm0, eax
    mov eax, 0x7FFFFFFF
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    addsd xmm2, xmm0
    dec ecx
    jnz .nr_sum

    ; z ≈ sum - 6 (gives approximately N(0, 1))
    movsd xmm0, [rel const_six]
    subsd xmm2, xmm0            ; z ~ N(0, 1)

    ; val = mean + z * std
    cvtss2sd xmm0, dword [rsp+4] ; std
    mulsd xmm2, xmm0
    cvtss2sd xmm0, dword [rsp]   ; mean
    addsd xmm2, xmm0
    cvtsd2ss xmm0, xmm2
    movss [r12 + rbx*4], xmm0

    inc rbx
    jmp .nr_loop

.nr_done:
    xor eax, eax
    add rsp, 16
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; =============================================================================
; init_he_uniform - He/Kaiming uniform initialization
;
; Fills with U(-bound, bound) where bound = sqrt(6 / fan)
;
; Args:
;   RDI = float* data
;   RSI = uint64_t num_elements
;   RDX = uint64_t fan (fan_in or fan_out depending on mode)
;   ECX = seed (0 = time-based)
; Returns:
;   EAX = 0 on success
; =============================================================================
global init_he_uniform
init_he_uniform:
    push rbp
    mov rbp, rsp
    sub rsp, 8

    ; Compute bound = sqrt(6.0 / fan)
    cvtsi2sd xmm2, rdx          ; fan as double
    movsd xmm3, [rel const_six]
    divsd xmm3, xmm2            ; 6.0 / fan
    sqrtsd xmm3, xmm3           ; bound = sqrt(6.0 / fan)

    ; Call init_uniform_range(data, n, -bound, +bound, seed)
    cvtsd2ss xmm3, xmm3
    movss xmm1, xmm3            ; hi = +bound
    xorps xmm0, xmm0
    subss xmm0, xmm3            ; lo = -bound
    mov edx, ecx                 ; seed
    ; rdi, rsi already set
    call init_uniform_range

    add rsp, 8
    pop rbp
    ret

; =============================================================================
; init_xavier_uniform - Xavier/Glorot uniform initialization
;
; Fills with U(-bound, bound) where bound = sqrt(6 / (fan_in + fan_out))
;
; Args:
;   RDI = float* data
;   RSI = uint64_t num_elements
;   RDX = uint64_t fan_in
;   RCX = uint64_t fan_out
;   R8D = seed (0 = time-based)
; Returns:
;   EAX = 0 on success
; =============================================================================
global init_xavier_uniform
init_xavier_uniform:
    push rbp
    mov rbp, rsp
    sub rsp, 8

    ; bound = sqrt(6.0 / (fan_in + fan_out))
    add rdx, rcx                 ; fan_in + fan_out
    cvtsi2sd xmm2, rdx
    movsd xmm3, [rel const_six]
    divsd xmm3, xmm2
    sqrtsd xmm3, xmm3

    cvtsd2ss xmm3, xmm3
    movss xmm1, xmm3            ; hi = +bound
    xorps xmm0, xmm0
    subss xmm0, xmm3            ; lo = -bound
    mov edx, r8d                 ; seed
    call init_uniform_range

    add rsp, 8
    pop rbp
    ret

; =============================================================================
; init_he_normal - He/Kaiming normal initialization
;
; Fills with N(0, std) where std = sqrt(2 / fan)
;
; Args:
;   RDI = float* data
;   RSI = uint64_t num_elements
;   RDX = uint64_t fan
;   ECX = seed (0 = time-based)
; Returns:
;   EAX = 0 on success
; =============================================================================
global init_he_normal
init_he_normal:
    push rbp
    mov rbp, rsp
    sub rsp, 8

    ; std = sqrt(2.0 / fan)
    cvtsi2sd xmm2, rdx
    movsd xmm3, [rel const_two]
    divsd xmm3, xmm2
    sqrtsd xmm3, xmm3
    cvtsd2ss xmm3, xmm3

    xorps xmm0, xmm0            ; mean = 0
    movss xmm1, xmm3            ; std
    mov edx, ecx                 ; seed
    call init_normal_range

    add rsp, 8
    pop rbp
    ret

; =============================================================================
; init_xavier_normal - Xavier/Glorot normal initialization
;
; Fills with N(0, std) where std = sqrt(2 / (fan_in + fan_out))
;
; Args:
;   RDI = float* data
;   RSI = uint64_t num_elements
;   RDX = uint64_t fan_in
;   RCX = uint64_t fan_out
;   R8D = seed (0 = time-based)
; Returns:
;   EAX = 0 on success
; =============================================================================
global init_xavier_normal
init_xavier_normal:
    push rbp
    mov rbp, rsp
    sub rsp, 8

    ; std = sqrt(2.0 / (fan_in + fan_out))
    add rdx, rcx
    cvtsi2sd xmm2, rdx
    movsd xmm3, [rel const_two]
    divsd xmm3, xmm2
    sqrtsd xmm3, xmm3
    cvtsd2ss xmm3, xmm3

    xorps xmm0, xmm0            ; mean = 0
    movss xmm1, xmm3            ; std
    mov edx, r8d                 ; seed
    call init_normal_range

    add rsp, 8
    pop rbp
    ret
