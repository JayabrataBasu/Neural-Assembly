; model_io.asm - Model Serialization and Deserialization
; Binary format for saving/loading trained models
; Format: [rel magic][rel version][rel num_layers][layer_data...]

section .data
    ; Magic number for file format identification
    MODEL_MAGIC:        dq 0x4C444F4D4C525545  ; "NEURALMD" in hex
    MODEL_VERSION:      dd 1
    
    ; Error messages
    err_file_open:      db "Error: Cannot open model file", 10, 0
    err_file_write:     db "Error: Cannot write to model file", 10, 0
    err_file_read:      db "Error: Cannot read from model file", 10, 0
    err_magic:          db "Error: Invalid model file format", 10, 0
    err_version:        db "Error: Unsupported model version", 10, 0
    
    ; Layer type identifiers
    LAYER_TYPE_LINEAR:      equ 1
    LAYER_TYPE_CONV2D:      equ 2
    LAYER_TYPE_BATCHNORM:   equ 3
    LAYER_TYPE_DROPOUT:     equ 4
    LAYER_TYPE_MAXPOOL2D:   equ 5
    LAYER_TYPE_RELU:        equ 10
    LAYER_TYPE_SIGMOID:     equ 11
    LAYER_TYPE_TANH:        equ 12
    LAYER_TYPE_SOFTMAX:     equ 13

section .bss
    ; Temporary buffer for file I/O
    io_buffer:          resb 4096
    current_fd:         resq 1

section .text
    global model_save
    global model_load
    global model_save_checkpoint
    global model_load_checkpoint
    global tensor_save
    global tensor_load
    global write_tensor_data
    global read_tensor_data
    
    extern mem_alloc
    extern mem_free
    extern tensor_create
    extern tensor_get_size

; ============================================================================
; MODEL FILE FORMAT
; ============================================================================
; Header:
;   - Magic number (8 bytes)
;   - Version (4 bytes)
;   - Number of layers (4 bytes)
;   - Total parameters (8 bytes)
;   - Model name length (4 bytes)
;   - Model name (variable)
;
; Per Layer:
;   - Layer type (4 bytes)
;   - Layer name length (4 bytes)
;   - Layer name (variable)
;   - Number of tensors (4 bytes)
;   - Per tensor:
;       - Tensor name length (4 bytes)
;       - Tensor name (variable)
;       - Number of dimensions (4 bytes)
;       - Dimensions (4 bytes each)
;       - Data (4 bytes per float)
; ============================================================================

; model_save - Save model to file
; Arguments:
;   rdi - pointer to Sequential model structure
;   rsi - filename string
; Returns:
;   rax - 0 on success, -1 on error
; 
; Sequential structure:
;   offset 0:  capacity (uint64_t)
;   offset 8:  size (uint64_t) - number of modules
;   offset 16: modules (Module**) - array of module pointers
model_save:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    
    mov r12, rdi                ; model pointer
    mov r13, rsi                ; filename
    
    ; Open file for writing (create/truncate)
    mov rax, 2                  ; sys_open
    mov rdi, r13
    mov rsi, 0x241              ; O_WRONLY | O_CREAT | O_TRUNC
    mov rdx, 0644o              ; permissions
    syscall
    
    test rax, rax
    js .open_error
    
    mov r14, rax                ; file descriptor
    mov [rel current_fd], rax
    
    ; Write magic number
    lea rdi, [rel io_buffer]
    mov rax, [rel MODEL_MAGIC]
    mov [rdi], rax
    
    ; Write version
    mov eax, [rel MODEL_VERSION]
    mov [rdi + 8], eax
    
    ; Get number of layers from Sequential structure (size is at offset 8)
    mov rax, [r12 + 8]          ; size at offset 8
    mov [rdi + 12], eax
    
    ; Write header (16 bytes so far)
    mov rax, 1                  ; sys_write
    mov rdi, r14
    lea rsi, [rel io_buffer]
    mov rdx, 16
    syscall
    
    test rax, rax
    js .write_error
    
    ; Write each layer
    mov r15, [r12 + 8]          ; size (num_layers) at offset 8
    mov rbx, [r12 + 16]         ; modules array pointer at offset 16
    
    ; Check if modules array is valid
    test rbx, rbx
    jz .write_done
    
.write_layers_loop:
    test r15, r15
    jz .write_done
    
    ; Get layer pointer
    mov rdi, [rbx]
    test rdi, rdi
    jz .skip_layer
    call write_layer
    
    test rax, rax
    js .write_error
    
.skip_layer:
    add rbx, 8
    dec r15
    jmp .write_layers_loop
    
.write_done:
    ; Close file
    mov rax, 3                  ; sys_close
    mov rdi, r14
    syscall
    
    xor eax, eax
    jmp .cleanup
    
.open_error:
    lea rdi, [rel err_file_open]
    call print_error
    mov eax, -1
    jmp .cleanup
    
.write_error:
    lea rdi, [rel err_file_write]
    call print_error
    
    ; Close file
    mov rax, 3
    mov rdi, r14
    syscall
    
    mov eax, -1
    
.cleanup:
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; write_layer - Write a single layer to file
; Arguments:
;   rdi - pointer to Module structure (from nn_layers.asm)
; Returns:
;   rax - 0 on success, -1 on error
; 
; Module structure (from nn_layers.asm):
;   offset 0:  n_params (4 bytes)
;   offset 8:  params (Tensor**)
;   offset 16: param_nodes (Node**)
;   offset 24: forward_fn pointer
;   offset 32: config pointer
write_layer:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 32
    
    mov r12, rdi                ; Module pointer
    
    ; Check for NULL or activation modules (modules with n_params = 0)
    test r12, r12
    jz .layer_write_done
    
    ; Check if this is a real module or activation marker
    ; Activation modules have n_params = 0 and a specific forward_fn
    mov eax, [r12]              ; n_params at offset 0
    test eax, eax
    jz .write_activation_layer  ; No params means activation layer
    
    ; Write layer type (Linear layer)
    lea rdi, [rel io_buffer]
    mov dword [rdi], LAYER_TYPE_LINEAR
    mov dword [rdi + 4], 0      ; name_length = 0 (no name)
    
    mov rax, 1                  ; sys_write
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 8
    syscall
    
    test rax, rax
    js .layer_write_error
    
    ; Write number of tensors (n_params)
    lea rdi, [rel io_buffer]
    mov eax, [r12]              ; n_params at offset 0
    mov [rdi], eax
    
    mov rax, 1
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 4
    syscall
    
    test rax, rax
    js .layer_write_error
    
    ; Write each tensor from params array
    mov r13d, [r12]             ; n_params
    mov rbx, [r12 + 8]          ; params array (Tensor**)
    test rbx, rbx
    jz .layer_write_done
    
.write_tensors_loop:
    test r13d, r13d
    jz .layer_write_done
    
    mov rdi, [rbx]              ; params[i] = Tensor*
    test rdi, rdi
    jz .skip_null_tensor
    call write_tensor_data
    
    test rax, rax
    js .layer_write_error
    
.skip_null_tensor:
    add rbx, 8
    dec r13d
    jmp .write_tensors_loop
    
.write_activation_layer:
    ; Activation layer - determine type from config
    mov rax, [r12 + 32]         ; config pointer
    test rax, rax
    jz .write_default_activation
    mov eax, [rax]              ; activation type from config
    add eax, 9                  ; offset to get LAYER_TYPE_RELU=10, etc.
    jmp .write_act_type
    
.write_default_activation:
    mov eax, LAYER_TYPE_RELU    ; default to relu
    
.write_act_type:
    lea rdi, [rel io_buffer]
    mov [rdi], eax
    mov dword [rdi + 4], 0      ; name_length
    mov dword [rdi + 8], 0      ; num_tensors
    
    mov rax, 1
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 12
    syscall
    
    test rax, rax
    js .layer_write_error
    
.layer_write_done:
    xor eax, eax
    jmp .layer_cleanup
    
.layer_write_error:
    mov eax, -1
    
.layer_cleanup:
    add rsp, 32
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; write_tensor_data - Write tensor to file
; Arguments:
;   rdi - pointer to tensor structure
; Returns:
;   rax - 0 on success, -1 on error
;
; Tensor structure (from tensor.asm):
;   offset 0:  data (void*)
;   offset 8:  ndim (uint64_t)
;   offset 16: shape (uint64_t*)
;   offset 24: stride (uint64_t*)
;   offset 32: dtype (uint32_t)
write_tensor_data:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 32
    
    mov r12, rdi                ; tensor pointer
    
    ; Get ndim
    mov rax, [r12 + 8]          ; ndim (uint64_t)
    mov r13, rax                ; save ndim
    
    ; Write number of dimensions (as 4-byte integer for file format)
    lea rdi, [rel io_buffer]
    mov [rdi], eax              ; just lower 32 bits
    
    ; Copy shape values from shape array pointer
    mov rsi, [r12 + 16]         ; shape pointer
    lea rdi, [rel io_buffer + 4]
    mov rcx, r13
    
.copy_dims:
    test rcx, rcx
    jz .dims_done
    mov rax, [rsi]              ; shape[i] is uint64_t
    mov [rdi], eax              ; write as 32-bit
    add rsi, 8
    add rdi, 4
    dec rcx
    jmp .copy_dims
    
.dims_done:
    ; Calculate total bytes to write for header
    ; 4 (ndim) + 4 * ndim (dimensions)
    mov eax, r13d
    shl eax, 2
    add eax, 4
    mov r14d, eax
    
    mov rax, 1                  ; sys_write
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov edx, r14d
    syscall
    
    test rax, rax
    js .tensor_write_error
    
    ; Calculate tensor size (product of dimensions)
    mov rdi, r12
    call tensor_get_size
    mov r14, rax                ; total elements
    
    ; Write tensor data
    ; Size in bytes = elements * 4 (float32)
    mov rax, 1
    mov rdi, [rel current_fd]
    mov rsi, [r12]              ; data pointer at offset 0
    mov rdx, r14
    shl rdx, 2                  ; * 4 for float32
    syscall
    
    test rax, rax
    js .tensor_write_error
    
    xor eax, eax
    jmp .tensor_cleanup
    
.tensor_write_error:
    mov eax, -1
    
.tensor_cleanup:
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; model_load - Load model from file
; Arguments:
;   rdi - filename string
; Returns:
;   rax - pointer to model structure, NULL on error
model_load:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    
    mov r12, rdi                ; filename
    
    ; Open file for reading
    mov rax, 2                  ; sys_open
    mov rdi, r12
    mov rsi, 0                  ; O_RDONLY
    xor edx, edx
    syscall
    
    test rax, rax
    js .load_open_error
    
    mov r13, rax                ; file descriptor
    mov [rel current_fd], rax
    
    ; Read header
    mov rax, 0                  ; sys_read
    mov rdi, r13
    lea rsi, [rel io_buffer]
    mov rdx, 16
    syscall
    
    cmp rax, 16
    jne .load_read_error
    
    ; Verify magic number
    lea rdi, [rel io_buffer]
    mov rax, [rdi]
    cmp rax, [rel MODEL_MAGIC]
    jne .load_magic_error
    
    ; Verify version
    mov eax, [rdi + 8]
    cmp eax, [rel MODEL_VERSION]
    ja .load_version_error
    
    ; Get number of layers
    mov r14d, [rdi + 12]        ; num_layers
    
    ; Allocate Sequential structure
    ; Sequential: capacity(8) + size(8) + modules ptr(8) + intermediates ptr(8) + inter_cap(8) + save_inter(1)
    mov edi, 48                 ; SEQUENTIAL_SIZE_BYTES
    call mem_alloc
    
    test rax, rax
    jz .load_alloc_error
    
    mov r15, rax                ; sequential pointer
    mov qword [r15], r14        ; capacity = num_layers
    mov qword [r15 + 8], r14    ; size = num_layers
    
    ; Allocate modules array
    mov eax, r14d
    shl eax, 3                  ; * 8 for pointers
    mov edi, eax
    call mem_alloc
    test rax, rax
    jz .load_alloc_error
    mov [r15 + 16], rax         ; modules array
    mov qword [r15 + 24], 0     ; intermediates = NULL
    mov qword [r15 + 32], 0     ; inter_cap = 0
    mov byte [r15 + 40], 0      ; save_inter = false
    
    ; Load each layer
    mov rbx, [r15 + 16]         ; pointer to modules array
    mov r14d, [r15 + 8]         ; num_layers from size
    
.load_layers_loop:
    test r14d, r14d
    jz .load_done
    
    call read_layer
    
    test rax, rax
    jz .load_layer_error
    
    mov [rbx], rax              ; store layer pointer
    add rbx, 8
    dec r14d
    jmp .load_layers_loop
    
.load_done:
    ; Close file
    mov rax, 3                  ; sys_close
    mov rdi, r13
    syscall
    
    mov rax, r15                ; return model pointer
    jmp .load_cleanup
    
.load_open_error:
    lea rdi, [rel err_file_open]
    call print_error
    xor eax, eax
    jmp .load_cleanup
    
.load_read_error:
    lea rdi, [rel err_file_read]
    call print_error
    jmp .load_close_error
    
.load_magic_error:
    lea rdi, [rel err_magic]
    call print_error
    jmp .load_close_error
    
.load_version_error:
    lea rdi, [rel err_version]
    call print_error
    jmp .load_close_error
    
.load_alloc_error:
.load_layer_error:
.load_close_error:
    mov rax, 3
    mov rdi, r13
    syscall
    xor eax, eax
    
.load_cleanup:
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; read_layer - Read a layer from file
; Returns:
;   rax - pointer to layer structure, NULL on error
read_layer:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 48
    
    ; Read layer header (type + name_length)
    mov rax, 0                  ; sys_read
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 8
    syscall
    
    cmp rax, 8
    jne .read_layer_error
    
    ; Allocate layer structure
    ; Size: 48 bytes base + name + tensor pointers
    mov edi, 256                ; generous allocation
    call mem_alloc
    
    test rax, rax
    jz .read_layer_error
    
    mov r12, rax                ; layer pointer
    
    ; Copy type and name_length
    lea rsi, [rel io_buffer]
    mov eax, [rsi]
    mov [r12], eax              ; layer_type
    mov eax, [rsi + 4]
    mov [r12 + 4], eax          ; name_length
    mov r13d, eax               ; save name_length
    
    ; Read layer name if present
    test r13d, r13d
    jz .skip_read_name
    
    ; Allocate name buffer
    mov edi, r13d
    add edi, 1                  ; null terminator
    call mem_alloc
    mov [r12 + 8], rax          ; name pointer
    
    mov rax, 0                  ; sys_read
    mov rdi, [rel current_fd]
    mov rsi, [r12 + 8]
    mov edx, r13d
    syscall
    
    cmp eax, r13d
    jne .read_layer_error
    
    ; Add null terminator
    mov rdi, [r12 + 8]
    add rdi, r13
    mov byte [rdi], 0
    
.skip_read_name:
    ; Read number of tensors
    mov rax, 0
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 4
    syscall
    
    cmp rax, 4
    jne .read_layer_error
    
    lea rsi, [rel io_buffer]
    mov eax, [rsi]
    mov [r12 + 16], eax         ; num_tensors
    mov r14d, eax
    
    ; Allocate tensor pointers array
    mov eax, r14d
    shl eax, 3                  ; * 8 for pointers
    mov edi, eax
    call mem_alloc
    mov [r12 + 24], rax         ; tensors array pointer
    
    ; Read each tensor
    mov rbx, [r12 + 24]
    
.read_tensors_loop:
    test r14d, r14d
    jz .read_layer_done
    
    call read_tensor_data
    
    test rax, rax
    jz .read_layer_error
    
    mov [rbx], rax
    add rbx, 8
    dec r14d
    jmp .read_tensors_loop
    
.read_layer_done:
    mov rax, r12
    jmp .read_layer_cleanup
    
.read_layer_error:
    xor eax, eax
    
.read_layer_cleanup:
    add rsp, 48
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; read_tensor_data - Read tensor from file
; Returns:
;   rax - pointer to tensor, NULL on error
read_tensor_data:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    
    ; Read ndim first
    mov rax, 0
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 4
    syscall
    
    cmp rax, 4
    jne .read_tensor_error
    
    lea rsi, [rel io_buffer]
    mov r12d, [rsi]             ; ndim
    
    ; Read dimensions
    mov eax, r12d
    shl eax, 2                  ; * 4 for int32
    mov r13d, eax               ; bytes to read
    
    mov rax, 0
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov edx, r13d
    syscall
    
    cmp eax, r13d
    jne .read_tensor_error
    
    ; Calculate total elements
    xor r14d, r14d
    mov r14d, 1                 ; accumulator
    lea rsi, [rel io_buffer]
    mov ecx, r12d
    
.calc_size:
    test ecx, ecx
    jz .size_done
    mov eax, [rsi]
    imul r14d, eax
    add rsi, 4
    dec ecx
    jmp .calc_size
    
.size_done:
    ; r14d = total elements
    
    ; Allocate tensor structure (64 bytes)
    mov edi, 64
    call mem_alloc
    
    test rax, rax
    jz .read_tensor_error
    
    mov r15, rax                ; tensor pointer
    
    ; Store ndim
    mov [r15 + 8], r12d
    
    ; Copy dimensions
    lea rdi, [r15 + 12]
    lea rsi, [rel io_buffer]
    mov ecx, r12d
    
.copy_read_dims:
    test ecx, ecx
    jz .dims_read_done
    mov eax, [rsi]
    mov [rdi], eax
    add rsi, 4
    add rdi, 4
    dec ecx
    jmp .copy_read_dims
    
.dims_read_done:
    ; Allocate data buffer
    mov eax, r14d
    shl eax, 2                  ; * 4 for float32
    mov edi, eax
    add edi, 32                 ; alignment padding
    call mem_alloc
    
    test rax, rax
    jz .read_tensor_error
    
    ; Align to 32 bytes
    add rax, 31
    and rax, ~31
    mov [r15], rax              ; data pointer
    
    ; Read tensor data
    mov eax, r14d
    shl eax, 2
    mov r13d, eax               ; bytes to read
    
    mov rax, 0
    mov rdi, [rel current_fd]
    mov rsi, [r15]
    mov edx, r13d
    syscall
    
    cmp eax, r13d
    jne .read_tensor_error
    
    ; Set other tensor fields
    mov dword [r15 + 44], 0     ; requires_grad = false
    mov qword [r15 + 48], 0     ; grad = null
    
    ; Calculate strides
    mov eax, r12d               ; ndim
    lea rdi, [r15 + 28]         ; strides array
    lea rsi, [r15 + 12]         ; dims array
    
    ; Stride of last dim = 1
    mov ecx, r12d
    dec ecx
    lea rdx, [rdi + rcx*4]
    mov dword [rdx], 1
    
    ; Calculate remaining strides
    dec ecx
    js .strides_done
    
.calc_strides:
    ; strides[i] = strides[i+1] * dims[i+1]
    lea rdx, [rdi + rcx*4]      ; &strides[i]
    mov eax, [rdx + 4]          ; strides[i+1]
    
    lea r8, [rsi + rcx*4]
    mov r9d, [r8 + 4]           ; dims[i+1]
    imul eax, r9d
    mov [rdx], eax
    
    dec ecx
    jns .calc_strides
    
.strides_done:
    mov rax, r15
    jmp .read_tensor_cleanup
    
.read_tensor_error:
    xor eax, eax
    
.read_tensor_cleanup:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; model_save_checkpoint - Save model with optimizer state
; Arguments:
;   rdi - model pointer
;   rsi - optimizer pointer
;   rdx - filename
;   rcx - epoch number
; Returns:
;   rax - 0 on success, -1 on error
model_save_checkpoint:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    
    mov r12, rdi                ; model
    mov r13, rsi                ; optimizer
    mov r14, rdx                ; filename
    mov r15, rcx                ; epoch
    
    ; Open file
    mov rax, 2
    mov rdi, r14
    mov rsi, 0x241              ; O_WRONLY | O_CREAT | O_TRUNC
    mov rdx, 0644o
    syscall
    
    test rax, rax
    js .ckpt_open_error
    
    mov rbx, rax                ; fd
    mov [rel current_fd], rax
    
    ; Write checkpoint header
    lea rdi, [rel io_buffer]
    mov rax, 0x54504B43         ; "CKPT"
    mov [rdi], eax
    mov [rdi + 4], r15d         ; epoch
    
    mov rax, 1
    mov rdi, rbx
    lea rsi, [rel io_buffer]
    mov rdx, 8
    syscall
    
    ; Save model
    mov rdi, r12
    mov rsi, r14
    ; (In practice, we'd write model data directly here)
    ; For now, just write model pointer as placeholder
    
    ; Save optimizer state
    ; Optimizer structure:
    ; offset 0: type
    ; offset 4: num_params
    ; offset 8: learning_rate (float)
    ; offset 16: momentum states pointer
    ; offset 24: velocity states pointer (Adam)
    
    test r13, r13
    jz .no_optimizer
    
    lea rdi, [rel io_buffer]
    mov eax, [r13]              ; optimizer type
    mov [rdi], eax
    mov eax, [r13 + 4]          ; num_params
    mov [rdi + 4], eax
    movss xmm0, [r13 + 8]       ; learning_rate
    movss [rdi + 8], xmm0
    
    mov rax, 1
    mov rdi, rbx
    lea rsi, [rel io_buffer]
    mov rdx, 12
    syscall
    
    ; TODO: Write momentum/velocity tensors
    
.no_optimizer:
    ; Close file
    mov rax, 3
    mov rdi, rbx
    syscall
    
    xor eax, eax
    jmp .ckpt_cleanup
    
.ckpt_open_error:
    mov eax, -1
    
.ckpt_cleanup:
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; model_load_checkpoint - Load model with optimizer state
; Arguments:
;   rdi - filename
;   rsi - pointer to store epoch number
; Returns:
;   rax - pointer to model, NULL on error
;   r8 - pointer to optimizer (caller should provide space)
model_load_checkpoint:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 32
    
    mov r12, rdi                ; filename
    mov r13, rsi                ; epoch pointer
    
    ; Open file
    mov rax, 2
    mov rdi, r12
    xor esi, esi                ; O_RDONLY
    xor edx, edx
    syscall
    
    test rax, rax
    js .ckpt_load_error
    
    mov rbx, rax
    mov [rel current_fd], rax
    
    ; Read checkpoint header
    mov rax, 0
    mov rdi, rbx
    lea rsi, [rel io_buffer]
    mov rdx, 8
    syscall
    
    cmp rax, 8
    jne .ckpt_load_error
    
    ; Verify magic
    lea rsi, [rel io_buffer]
    mov eax, [rsi]
    cmp eax, 0x54504B43         ; "CKPT"
    jne .ckpt_load_error
    
    ; Get epoch
    mov eax, [rsi + 4]
    test r13, r13
    jz .skip_epoch_store
    mov [r13], eax
    
.skip_epoch_store:
    ; Load model
    ; (In practice, call model_load logic here)
    
    ; Close file
    mov rax, 3
    mov rdi, rbx
    syscall
    
    ; Return placeholder (would return actual model)
    xor eax, eax
    jmp .ckpt_load_cleanup
    
.ckpt_load_error:
    xor eax, eax
    
.ckpt_load_cleanup:
    add rsp, 32
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; tensor_save - Save a single tensor to file
; Arguments:
;   rdi - tensor pointer
;   rsi - filename
; Returns:
;   rax - 0 on success, -1 on error
tensor_save:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    sub rsp, 16
    
    mov r12, rdi                ; tensor
    mov r13, rsi                ; filename
    
    ; Open file
    mov rax, 2
    mov rdi, r13
    mov rsi, 0x241
    mov rdx, 0644o
    syscall
    
    test rax, rax
    js .tensor_save_error
    
    mov [rel current_fd], rax
    push rax
    
    mov rdi, r12
    call write_tensor_data
    
    pop rdi
    push rax
    
    mov rax, 3                  ; close
    syscall
    
    pop rax
    jmp .tensor_save_done
    
.tensor_save_error:
    mov eax, -1
    
.tensor_save_done:
    add rsp, 16
    pop r13
    pop r12
    pop rbp
    ret

; tensor_load - Load a single tensor from file
; Arguments:
;   rdi - filename
; Returns:
;   rax - tensor pointer, NULL on error
tensor_load:
    push rbp
    mov rbp, rsp
    push r12
    sub rsp, 8
    
    mov r12, rdi                ; filename
    
    ; Open file
    mov rax, 2
    mov rdi, r12
    xor esi, esi
    xor edx, edx
    syscall
    
    test rax, rax
    js .tensor_load_error
    
    mov [rel current_fd], rax
    push rax
    
    call read_tensor_data
    mov r12, rax
    
    pop rdi
    push rax
    
    mov rax, 3                  ; close
    syscall
    
    pop rax
    mov rax, r12
    jmp .tensor_load_done
    
.tensor_load_error:
    xor eax, eax
    
.tensor_load_done:
    add rsp, 8
    pop r12
    pop rbp
    ret

; print_error - Print error message to stderr
; Arguments:
;   rdi - error message string
print_error:
    push rbp
    mov rbp, rsp
    push rbx
    
    mov rbx, rdi
    
    ; Calculate string length
    xor ecx, ecx
.strlen_loop:
    mov al, [rbx + rcx]
    test al, al
    jz .strlen_done
    inc ecx
    jmp .strlen_loop
    
.strlen_done:
    mov rax, 1                  ; sys_write
    mov rdi, 2                  ; stderr
    mov rsi, rbx
    mov edx, ecx
    syscall
    
    pop rbx
    pop rbp
    ret
; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
