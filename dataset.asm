; =============================================================================
; dataset.asm - Dataset Loading and Batching
; =============================================================================
; CSV loading, batch iteration
; =============================================================================

; Dataset struct layout (48 bytes):
; Offset  Size    Field
; 0       8       n_samples    (uint64_t)
; 8       8       data         (Tensor*) - features
; 16      8       labels       (Tensor*) - labels/targets
; 24      8       n_features   (uint64_t)
; 32      8       n_classes    (uint64_t) - for classification
; 40      8       dtype        (uint32_t)

%define DATASET_SIZE        48
%define DATASET_N_SAMPLES   0
%define DATASET_DATA        8
%define DATASET_LABELS      16
%define DATASET_N_FEATURES  24
%define DATASET_N_CLASSES   32
%define DATASET_DTYPE       40

%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_DTYPE        32

%define DT_FLOAT32          0
%define DT_FLOAT64          1

section .data
    align 8
    err_file_open:      db "Error: Could not open file", 0
    err_parse:          db "Error: CSV parse error", 0
    mode_read:          db "r", 0
    csv_delim:          db ",", 0
    newline_char:       db 10, 0
    dbg_samples:        db "[DBG] n_samples = ", 0
    dbg_features:       db "[DBG] n_features = ", 0
    dbg_nl:             db 10, 0
    dbg_label_path:     db "[DBG] Label path: ", 0
    dbg_line_content:   db "[DBG] Line content: ", 0
    dbg_float_val:      db "[DBG] Float val: ", 0
    dbg_dtype:          db "[DBG] dtype: ", 0
    dbg_storing_f32:    db "[DBG] Storing f32", 0
    dbg_labels_tensor_val: db "[DBG] Labels tensor[0]: ", 0
    dbg_batch_y_val:    db "[DBG] Batch Y[0]: ", 0

section .bss
    align 8
    line_buffer:        resb 4096
    temp_values:        resq 1024

section .text

; External libc functions
extern fopen
extern fclose
extern fgets
extern strtok
extern strtod
extern atoi
extern atol

; External functions
extern mem_alloc
extern mem_alloc_aligned
extern mem_free
extern tensor_create
extern tensor_zeros
extern tensor_free
extern tensor_numel
extern panic
extern str_to_float
extern str_to_int
extern print_string
extern print_int
extern print_float

; Export dataset functions
global dataset_create
global dataset_load_csv
global dataset_free
global dataset_get_batch
global dataset_shuffle_indices

; =============================================================================
; dataset_create - Create empty dataset
; Arguments:
;   RDI = n_samples (uint64_t)
;   RSI = n_features (uint64_t)
;   RDX = dtype (uint32_t)
; Returns:
;   RAX = Dataset*
; =============================================================================
dataset_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 32
    
    mov r12, rdi                    ; n_samples
    mov r13, rsi                    ; n_features
    mov r14d, edx                   ; dtype
    
    ; Allocate dataset struct
    mov rdi, DATASET_SIZE
    call mem_alloc
    test rax, rax
    jz .alloc_failed
    mov rbx, rax
    
    ; Initialize
    mov [rbx + DATASET_N_SAMPLES], r12
    mov [rbx + DATASET_N_FEATURES], r13
    mov [rbx + DATASET_DTYPE], r14d
    mov qword [rbx + DATASET_N_CLASSES], 0
    
    ; Create data tensor (n_samples x n_features)
    mov [rsp], r12                  ; shape[0]
    mov [rsp+8], r13                ; shape[1]
    mov rdi, 2
    lea rsi, [rsp]
    mov edx, r14d
    call tensor_create
    mov [rbx + DATASET_DATA], rax
    
    ; Labels tensor created later when loading
    mov qword [rbx + DATASET_LABELS], 0
    
    mov rax, rbx
    
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

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
; dataset_load_csv - Load dataset from CSV files
; Arguments:
;   RDI = const char* data_path
;   RSI = const char* label_path
;   RDX = n_features (uint64_t)
;   RCX = dtype (uint32_t)
; Returns:
;   RAX = Dataset*
; =============================================================================
dataset_load_csv:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 88                     ; Align stack (5 pushes = 40 bytes, need 8 more + 80 = 128 bytes aligned)
    
    mov [rsp], rdi                  ; data_path
    mov [rsp+8], rsi                ; label_path
    mov [rsp+16], rdx               ; n_features
    mov [rsp+24], ecx               ; dtype
    
    ; First pass: count lines in data file
    mov rdi, [rsp]
    lea rsi, [rel mode_read]
    call fopen wrt ..plt
    test rax, rax
    jz .file_error
    mov [rsp+32], rax               ; data file handle
    
    xor r12, r12                    ; line count
.count_lines:
    mov rdi, [rsp+32]
    lea rsi, [rel line_buffer]
    mov edx, 4096
    mov rcx, rdi
    mov rdi, rsi
    mov rsi, rdx
    mov rdx, rcx
    call fgets wrt ..plt
    test rax, rax
    jz .done_counting
    inc r12
    jmp .count_lines

.done_counting:
    mov rdi, [rsp+32]
    call fclose wrt ..plt
    
    mov [rsp+40], r12               ; n_samples
    
    ; Allocate dataset
    mov rdi, DATASET_SIZE
    call mem_alloc
    test rax, rax
    jz .alloc_failed
    mov r13, rax                    ; dataset
    
    mov rax, [rsp+40]
    mov [r13 + DATASET_N_SAMPLES], rax
    mov rax, [rsp+16]
    mov [r13 + DATASET_N_FEATURES], rax
    mov eax, [rsp+24]
    mov [r13 + DATASET_DTYPE], eax
    
    ; Create data tensor
    mov rax, [rsp+40]
    mov [rsp+48], rax               ; shape[0] = n_samples
    mov rax, [rsp+16]
    mov [rsp+56], rax               ; shape[1] = n_features
    
    mov rdi, 2
    lea rsi, [rsp+48]
    mov edx, [rsp+24]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov [r13 + DATASET_DATA], rax
    mov r14, rax                    ; data tensor
    
    ; Create labels tensor (1D, n_samples)
    mov rax, [rsp+40]
    mov [rsp+48], rax
    mov rdi, 1
    lea rsi, [rsp+48]
    mov edx, [rsp+24]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov [r13 + DATASET_LABELS], rax
    mov r15, rax                    ; labels tensor
    
    ; Re-open and parse data file
    mov rdi, [rsp]
    lea rsi, [rel mode_read]
    call fopen wrt ..plt
    test rax, rax
    jz .file_error
    mov [rsp+32], rax
    
    mov rbx, [r14 + TENSOR_DATA]    ; data pointer
    xor r12, r12                    ; sample index

.parse_data_lines:
    cmp r12, [rsp+40]
    jge .done_data
    
    ; Read line
    lea rdi, [rel line_buffer]
    mov esi, 4096
    mov rdx, [rsp+32]
    call fgets wrt ..plt
    test rax, rax
    jz .done_data
    
    ; Parse CSV line
    lea rdi, [rel line_buffer]
    lea rsi, [rel csv_delim]
    call strtok wrt ..plt
    
    xor rcx, rcx                    ; feature index
.parse_features:
    test rax, rax
    jz .next_sample
    cmp rcx, [rsp+16]
    jge .next_sample
    
    ; Convert to float
    push rcx
    push rbx
    mov rdi, rax
    xor esi, esi
    call strtod wrt ..plt
    pop rbx
    pop rcx
    
    ; Store value
    mov eax, [rsp+24]
    cmp eax, DT_FLOAT32
    je .store_f32
    
    ; float64
    mov rax, r12
    imul rax, [rsp+16]
    add rax, rcx
    movsd [rbx + rax*8], xmm0
    jmp .next_feature

.store_f32:
    cvtsd2ss xmm0, xmm0
    mov rax, r12
    imul rax, [rsp+16]
    add rax, rcx
    movss [rbx + rax*4], xmm0

.next_feature:
    inc rcx
    
    ; Get next token
    push rcx
    push rbx
    xor edi, edi
    lea rsi, [rel csv_delim]
    call strtok wrt ..plt
    pop rbx
    pop rcx
    jmp .parse_features

.next_sample:
    inc r12
    jmp .parse_data_lines

.done_data:
    mov rdi, [rsp+32]
    call fclose wrt ..plt
    
    ; Load labels file
    mov rax, [rsp+8]
    test rax, rax
    jz .no_labels
    
    mov rdi, rax
    lea rsi, [rel mode_read]
    call fopen wrt ..plt
    test rax, rax
    jz .labels_open_failed
    mov [rsp+32], rax
    
    mov rbx, [r15 + TENSOR_DATA]
    xor r12, r12

.parse_labels:
    cmp r12, [rsp+40]
    jge .done_labels
    
    lea rdi, [rel line_buffer]
    mov esi, 4096
    mov rdx, [rsp+32]
    call fgets wrt ..plt
    test rax, rax
    jz .done_labels
    
    ; Parse label (integer or float)
    push r12
    push rbx
    lea rdi, [rel line_buffer]
    xor esi, esi
    call strtod wrt ..plt
    pop rbx
    pop r12
    
    mov eax, [rsp+24]
    cmp eax, DT_FLOAT32
    je .store_label_f32
    
    movsd [rbx + r12*8], xmm0
    jmp .next_label

.store_label_f32:
    cvtsd2ss xmm0, xmm0
    movss [rbx + r12*4], xmm0

.next_label:
    inc r12
    jmp .parse_labels

.done_labels:
    mov rdi, [rsp+32]
    call fclose wrt ..plt

.no_labels:
    mov rax, r13
    
    add rsp, 88
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.labels_open_failed:
    lea rdi, [rel err_file_open]
    call print_string
    lea rdi, [rel dbg_nl]
    call print_string
    jmp .no_labels

.file_error:
    lea rdi, [rel err_file_open]
    call panic

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
; dataset_free - Free dataset
; Arguments:
;   RDI = Dataset* dataset
; =============================================================================
dataset_free:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8
    
    test rdi, rdi
    jz .done
    
    mov rbx, rdi
    
    ; Free data tensor
    mov rdi, [rbx + DATASET_DATA]
    test rdi, rdi
    jz .skip_data
    call tensor_free
.skip_data:
    
    ; Free labels tensor
    mov rdi, [rbx + DATASET_LABELS]
    test rdi, rdi
    jz .skip_labels
    call tensor_free
.skip_labels:
    
    ; Free dataset struct
    mov rdi, rbx
    call mem_free

.done:
    add rsp, 8
    pop rbx
    pop rbp
    ret

; =============================================================================
; dataset_get_batch - Get a batch of samples
; Arguments:
;   RDI = Dataset* dataset
;   RSI = batch_index (uint64_t)
;   RDX = batch_size (uint64_t)
;   RCX = Tensor** out_x (output)
;   R8 = Tensor** out_y (output)
; =============================================================================
dataset_get_batch:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; dataset
    mov r13, rsi                    ; batch_index
    mov r14, rdx                    ; batch_size
    mov [rsp], rcx                  ; out_x
    mov [rsp+8], r8                 ; out_y
    
    ; Calculate start index
    mov rax, r13
    imul rax, r14
    mov r15, rax                    ; start_idx
    
    ; Clamp batch_size to remaining samples
    mov rax, [r12 + DATASET_N_SAMPLES]
    sub rax, r15
    cmp r14, rax
    jle .size_ok
    mov r14, rax
.size_ok:
    
    ; Get data tensor info
    mov rax, [r12 + DATASET_DATA]
    mov rbx, rax                    ; data tensor
    mov rcx, [r12 + DATASET_N_FEATURES]
    mov [rsp+16], rcx               ; n_features
    mov ecx, [rax + TENSOR_DTYPE]
    mov [rsp+24], ecx               ; dtype
    
    ; Create batch X tensor (batch_size x n_features)
    mov [rsp+32], r14               ; shape[0]
    mov rax, [rsp+16]
    mov [rsp+40], rax               ; shape[1]
    
    mov rdi, 2
    lea rsi, [rsp+32]
    mov edx, [rsp+24]
    call tensor_create
    mov [rsp+48], rax               ; batch_x tensor
    
    ; Copy data
    mov rdi, [rax + TENSOR_DATA]    ; dst = batch_x->data
    mov rax, [rbx + TENSOR_DATA]    ; src base = data_tensor->data
    
    ; Calculate source offset
    mov rcx, r15                    ; start_idx
    imul rcx, [rsp+16]              ; * n_features
    
    mov esi, [rsp+24]
    cmp esi, DT_FLOAT32
    je .copy_x_f32
    
    ; float64
    shl rcx, 3                      ; * 8
    add rax, rcx                    ; src = data + start_idx * n_features * 8
    mov rsi, rax
    
    mov rcx, r14
    imul rcx, [rsp+16]              ; total elements = batch_size * n_features
    shl rcx, 3                      ; * 8 bytes
    ; rdi already has destination
    rep movsb
    jmp .copy_y

.copy_x_f32:
    shl rcx, 2                      ; * 4
    add rax, rcx                    ; src = data + start_idx * n_features * 4
    mov rsi, rax
    
    mov rcx, r14
    imul rcx, [rsp+16]              ; total elements = batch_size * n_features
    shl rcx, 2                      ; * 4 bytes
    ; rdi already has destination
    rep movsb

.copy_y:
    ; Store batch_x
    mov rax, [rsp]                  ; out_x
    mov rcx, [rsp+48]
    mov [rax], rcx
    
    ; Create batch Y tensor if labels exist
    mov rax, [r12 + DATASET_LABELS]
    test rax, rax
    jz .no_batch_labels
    
    mov rbx, rax                    ; labels tensor
    
    ; Create batch_y (batch_size,)
    mov [rsp+32], r14               ; shape[0]
    mov rdi, 1
    lea rsi, [rsp+32]
    mov edx, [rsp+24]
    call tensor_create
    mov [rsp+48], rax               ; batch_y tensor
    
    ; Copy labels
    mov rdi, [rax + TENSOR_DATA]
    mov rax, [rbx + TENSOR_DATA]
    
    mov rcx, r15                    ; start_idx
    mov esi, [rsp+24]
    cmp esi, DT_FLOAT32
    je .copy_y_f32
    
    shl rcx, 3
    add rax, rcx
    mov rsi, rax                    ; source = labels_data + start_idx * 8
    mov rcx, r14
    shl rcx, 3                      ; byte count = batch_size * 8
    ; rdi already has destination (batch_y->data)
    rep movsb
    jmp .store_y

.copy_y_f32:
    shl rcx, 2
    add rax, rcx
    mov rsi, rax                    ; source = labels_data + start_idx * 4
    mov rcx, r14
    shl rcx, 2                      ; byte count = batch_size * 4
    ; rdi already has destination (batch_y->data from line above)
    rep movsb

.store_y:
    mov rax, [rsp+8]                ; out_y
    mov rcx, [rsp+48]
    mov [rax], rcx
    jmp .done

.no_batch_labels:
    mov rax, [rsp+8]
    mov qword [rax], 0

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
; dataset_shuffle_indices - Create shuffled index array
; Arguments:
;   RDI = Dataset* dataset
;   RSI = uint64_t* index_array (output, must be pre-allocated)
; =============================================================================
dataset_shuffle_indices:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    mov r12, rdi                    ; dataset
    mov r13, rsi                    ; index array
    
    mov rbx, [r12 + DATASET_N_SAMPLES]
    
    ; Initialize indices 0, 1, 2, ...
    xor rcx, rcx
.init_loop:
    cmp rcx, rbx
    jge .shuffle
    mov [r13 + rcx*8], rcx
    inc rcx
    jmp .init_loop

.shuffle:
    ; Fisher-Yates shuffle (simple version using time as seed)
    ; For now, just leave as sequential (shuffling requires rand)
    ; TODO: Implement proper shuffling with random
    
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
