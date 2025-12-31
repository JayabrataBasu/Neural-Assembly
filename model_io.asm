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
    global linear_forward_loaded
    global activation_relu_forward
    global activation_sigmoid_forward
    global activation_tanh_forward
    global activation_softmax_forward
    
    extern mem_alloc
    extern mem_free
    extern mem_zero
    extern tensor_create
    extern tensor_free
    extern tensor_get_size
    extern print_error
    extern node_create
    extern matmul
    extern ew_add
    extern node_relu
    extern node_sigmoid
    extern node_tanh
    extern node_softmax

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

; read_layer - Read a layer from file and reconstruct proper Module structure
; Returns:
;   rax - pointer to Module structure, NULL on error
; 
; This function reads saved layer data and reconstructs a proper Module struct
; that is compatible with model_forward's expectations:
;   Module structure (64 bytes):
;     offset 0:  n_params (4 bytes)
;     offset 8:  params (Tensor**)
;     offset 16: param_nodes (Node**)
;     offset 24: forward_fn pointer
;     offset 32: config pointer
read_layer:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    ; Read layer header (type + name_length)
    mov rax, 0                  ; sys_read
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 8
    syscall
    
    cmp rax, 8
    jne .read_layer_error
    
    ; Get layer type
    lea rsi, [rel io_buffer]
    mov r13d, [rsi]             ; layer_type
    mov r14d, [rsi + 4]         ; name_length
    
    ; Skip layer name if present
    test r14d, r14d
    jz .skip_read_name
    
    ; Read and discard name
    mov rax, 0                  ; sys_read
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov edx, r14d
    syscall
    
    cmp eax, r14d
    jne .read_layer_error
    
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
    mov r15d, [rsi]             ; num_tensors
    
    ; Now create proper Module structure based on layer type
    ; Allocate Module structure (64 bytes)
    mov edi, 64
    call mem_alloc
    test rax, rax
    jz .read_layer_error
    mov r12, rax                ; Module pointer
    
    ; Zero out the module structure
    mov rdi, r12
    mov esi, 64
    call mem_zero
    
    ; Check layer type and reconstruct accordingly
    cmp r13d, LAYER_TYPE_LINEAR
    je .read_linear_layer
    
    ; Activation layer (RELU=10, SIGMOID=11, TANH=12, SOFTMAX=13)
    cmp r13d, LAYER_TYPE_RELU
    je .read_activation_layer
    cmp r13d, LAYER_TYPE_SIGMOID
    je .read_activation_layer
    cmp r13d, LAYER_TYPE_TANH
    je .read_activation_layer
    cmp r13d, LAYER_TYPE_SOFTMAX
    je .read_activation_layer
    
    ; Unknown layer type - treat as activation
    jmp .read_activation_layer
    
.read_linear_layer:
    ; Linear layer - read tensors and set up Module
    mov dword [r12], r15d       ; n_params = num_tensors (usually 2: weight, bias)
    
    ; Allocate params array (Tensor**)
    mov eax, r15d
    shl eax, 3                  ; * 8 for pointers
    mov edi, eax
    test edi, edi
    jz .linear_no_params
    call mem_alloc
    test rax, rax
    jz .read_layer_error
    mov [r12 + 8], rax          ; params array
    mov rbx, rax
    
    ; Read each tensor
    mov r14d, r15d              ; counter
.read_linear_tensors:
    test r14d, r14d
    jz .linear_tensors_done
    
    call read_tensor_data
    test rax, rax
    jz .read_layer_error
    
    mov [rbx], rax              ; store tensor pointer
    add rbx, 8
    dec r14d
    jmp .read_linear_tensors
    
.linear_tensors_done:
    ; IMPORTANT: Create param_nodes from loaded tensors
    ; linear_forward_fn expects param_nodes, not just params
    mov eax, [r12]              ; n_params
    test eax, eax
    jz .skip_param_nodes
    
    ; Allocate param_nodes array
    shl eax, 3                  ; * 8 for pointers
    mov edi, eax
    call mem_alloc
    test rax, rax
    jz .read_layer_error
    mov [r12 + 16], rax         ; param_nodes array
    mov rbx, rax                ; param_nodes array
    
    ; Create nodes from each tensor
    mov r14d, [r12]             ; n_params count
    mov r15, [r12 + 8]          ; params array
    
.create_param_nodes_loop:
    test r14d, r14d
    jz .skip_param_nodes
    
    mov rdi, [r15]              ; tensor
    mov rsi, 1                  ; requires_grad = true
    call node_create
    test rax, rax
    jz .read_layer_error
    
    mov [rbx], rax              ; store node
    add rbx, 8
    add r15, 8
    dec r14d
    jmp .create_param_nodes_loop
    
.skip_param_nodes:
    ; Set forward_fn (though for Linear with n_params > 0, 
    ; model_forward bypasses this and calls linear_forward directly)
    lea rax, [rel linear_forward_loaded]
    mov [r12 + 24], rax
    
    ; Create config with in/out features from weight tensor shape
    mov edi, 16                 ; config size
    call mem_alloc
    test rax, rax
    jz .read_layer_done_ok      ; config optional
    mov [r12 + 32], rax
    
    ; Get dimensions from weight tensor (shape: [out_features, in_features])
    mov rbx, [r12 + 8]          ; params array
    mov rbx, [rbx]              ; weight tensor
    test rbx, rbx
    jz .read_layer_done_ok
    mov rcx, [rbx + 16]         ; shape pointer (TENSOR_SHAPE=16)
    test rcx, rcx
    jz .read_layer_done_ok
    mov rdi, [r12 + 32]         ; config
    mov rax, [rcx + 8]          ; shape[1] = in_features (assuming shape is 64-bit)
    mov [rdi], rax
    mov rax, [rcx]              ; shape[0] = out_features
    mov [rdi + 8], rax
    jmp .read_layer_done_ok
    
.linear_no_params:
    mov qword [r12 + 8], 0
    lea rax, [rel linear_forward_loaded]
    mov [r12 + 24], rax
    jmp .read_layer_done_ok
    
.read_activation_layer:
    ; Activation layer - no params, set up forward_fn based on type
    mov dword [r12], 0          ; n_params = 0
    mov qword [r12 + 8], 0      ; params = NULL
    mov qword [r12 + 16], 0     ; param_nodes = NULL
    
    ; Allocate config for activation type
    mov edi, 16
    call mem_alloc
    test rax, rax
    jz .set_act_forward
    mov [r12 + 32], rax
    
    ; Convert LAYER_TYPE to activation type
    ; LAYER_TYPE_RELU=10 -> ACT_RELU=1, etc.
    mov eax, r13d
    sub eax, 9                  ; LAYER_TYPE_RELU(10) - 9 = 1 = ACT_RELU
    mov rdi, [r12 + 32]
    mov [rdi], eax              ; activation_type
    
.set_act_forward:
    ; Set forward_fn based on activation type
    cmp r13d, LAYER_TYPE_RELU
    je .set_relu_forward
    cmp r13d, LAYER_TYPE_SIGMOID
    je .set_sigmoid_forward
    cmp r13d, LAYER_TYPE_TANH
    je .set_tanh_forward
    cmp r13d, LAYER_TYPE_SOFTMAX
    je .set_softmax_forward
    ; Default to relu
    jmp .set_relu_forward
    
.set_relu_forward:
    lea rax, [rel activation_relu_forward]
    mov [r12 + 24], rax
    jmp .read_layer_done_ok
    
.set_sigmoid_forward:
    lea rax, [rel activation_sigmoid_forward]
    mov [r12 + 24], rax
    jmp .read_layer_done_ok
    
.set_tanh_forward:
    lea rax, [rel activation_tanh_forward]
    mov [r12 + 24], rax
    jmp .read_layer_done_ok
    
.set_softmax_forward:
    lea rax, [rel activation_softmax_forward]
    mov [r12 + 24], rax
    jmp .read_layer_done_ok
    
.read_layer_done_ok:
    mov rax, r12
    jmp .read_layer_cleanup
    
.read_layer_error:
    xor eax, eax
    
.read_layer_cleanup:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; read_tensor_data - Read tensor from file using proper tensor_create
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
    sub rsp, 80                 ; Extra space for shape array
    
    ; Read ndim first
    mov rax, 0
    mov rdi, [rel current_fd]
    lea rsi, [rel io_buffer]
    mov rdx, 4
    syscall
    
    cmp rax, 4
    jne .read_tensor_error
    
    lea rsi, [rel io_buffer]
    mov r12d, [rsi]             ; ndim (32-bit from file)
    mov [rbp - 56], r12         ; save ndim as 64-bit
    
    ; Read dimensions (32-bit integers from file)
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
    
    ; Convert 32-bit dimensions to 64-bit shape array on stack
    ; Store shape at [rbp - 48] (up to 4 dims * 8 bytes = 32 bytes)
    lea rdi, [rbp - 48]         ; destination for 64-bit shape
    lea rsi, [rel io_buffer]    ; source of 32-bit dims
    mov ecx, r12d               ; ndim
    xor r14, r14                ; total elements = 1
    mov r14d, 1
    
.convert_dims:
    test ecx, ecx
    jz .dims_converted
    
    mov eax, [rsi]              ; read 32-bit dim
    mov [rdi], rax              ; store as 64-bit (zero-extended)
    imul r14d, eax              ; accumulate total
    
    add rsi, 4
    add rdi, 8
    dec ecx
    jmp .convert_dims
    
.dims_converted:
    ; r14d = total elements
    mov [rbp - 64], r14         ; save total elements
    
    ; Now create tensor properly using tensor_create
    mov rdi, [rbp - 56]         ; ndim (64-bit)
    lea rsi, [rbp - 48]         ; shape array pointer (64-bit values)
    mov edx, 0                  ; dtype = DT_FLOAT32
    call tensor_create
    
    test rax, rax
    jz .read_tensor_error
    mov r15, rax                ; tensor pointer
    
    ; Read tensor data directly into tensor's data buffer
    mov r14, [rbp - 64]         ; total elements
    mov eax, r14d
    shl eax, 2                  ; * 4 for float32
    mov r13d, eax               ; bytes to read
    
    mov rax, 0                  ; sys_read
    mov rdi, [rel current_fd]
    mov rsi, [r15]              ; TENSOR_DATA (offset 0)
    mov edx, r13d
    syscall
    
    cmp eax, r13d
    jne .read_tensor_error_cleanup
    
    mov rax, r15                ; return tensor pointer
    jmp .read_tensor_cleanup
    
.read_tensor_error_cleanup:
    ; Free the tensor we allocated
    mov rdi, r15
    call tensor_free
    
.read_tensor_error:
    xor eax, eax
    
.read_tensor_cleanup:
    add rsp, 80
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

; =============================================================================
; Forward functions for loaded models
; These are simplified versions that work with loaded Module structures
; =============================================================================

; linear_forward_loaded - Forward pass for loaded Linear layer
; Arguments:
;   rdi - Module pointer (with params[0]=weight, params[1]=bias)
;   rsi - input Node
;   rdx - output Node pointer (for compatibility with activation signature)
; Returns:
;   rax - output Node (or 0 on error)
linear_forward_loaded:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    
    mov r12, rdi                ; Module
    mov r13, rsi                ; input node
    mov [rbp - 40], rdx         ; output pointer (optional)
    
    ; Get weight and bias tensors from params
    mov rax, [r12 + 8]          ; params array
    test rax, rax
    jz .linear_loaded_error
    
    mov r14, [rax]              ; weight tensor
    mov r15, [rax + 8]          ; bias tensor (may be NULL)
    
    ; Get input tensor from node
    mov rax, [r13]              ; NODE_VALUE = input tensor
    test rax, rax
    jz .linear_loaded_error
    mov rbx, rax                ; input tensor
    
    ; Compute output = input @ weight^T + bias
    ; For simplicity, create output tensor first
    ; Get output dimensions from weight: [out_features, in_features]
    mov rax, [r14 + 16]         ; weight shape pointer
    test rax, rax
    jz .linear_loaded_error
    
    ; Create output tensor with shape [batch_size, out_features]
    mov rcx, [rbx + 16]         ; input shape pointer
    test rcx, rcx
    jz .linear_loaded_error
    
    ; Get batch_size from input shape[0]
    mov r8, [rcx]               ; batch_size (or in_features if 1D)
    
    ; Get out_features from weight shape[0]
    mov r9, [rax]               ; out_features
    
    ; For now, do a simple matrix multiply
    ; output = matmul(input, weight^T)
    mov rdi, rbx                ; input tensor
    mov rsi, r14                ; weight tensor  
    call matmul
    
    test rax, rax
    jz .linear_loaded_error
    mov rbx, rax                ; output tensor
    
    ; Add bias if present
    test r15, r15
    jz .linear_loaded_no_bias
    
    mov rdi, rbx                ; output tensor
    mov rsi, r15                ; bias tensor
    call ew_add
    test rax, rax
    jz .linear_loaded_error
    mov rbx, rax
    
.linear_loaded_no_bias:
    ; Create output node
    mov rdi, rbx                ; output tensor
    mov rsi, 1                  ; requires_grad = true for training compatibility
    call node_create
    
    test rax, rax
    jz .linear_loaded_error
    
    ; Store output if pointer provided
    mov rdx, [rbp - 40]
    test rdx, rdx
    jz .linear_loaded_done
    mov [rdx], rax
    xor eax, eax                ; return 0 for success (activation convention)
    jmp .linear_loaded_cleanup
    
.linear_loaded_done:
    ; Return node directly
    jmp .linear_loaded_cleanup
    
.linear_loaded_error:
    xor eax, eax
    
.linear_loaded_cleanup:
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; activation_relu_forward - ReLU forward for loaded model
; Arguments:
;   rdi - Module pointer (unused)
;   rsi - input Node
;   rdx - output Node pointer
; Returns:
;   rax - 0 on success, -1 on error
activation_relu_forward:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8
    
    mov rbx, rdx                ; save output pointer
    mov rdi, rsi                ; input node
    call node_relu
    
    test rax, rax
    jz .relu_fwd_error
    
    mov [rbx], rax              ; store output node
    xor eax, eax                ; return 0 = success
    jmp .relu_fwd_done
    
.relu_fwd_error:
    mov eax, -1
    
.relu_fwd_done:
    add rsp, 8
    pop rbx
    pop rbp
    ret

; activation_sigmoid_forward - Sigmoid forward for loaded model
; Arguments:
;   rdi - Module pointer (unused)
;   rsi - input Node
;   rdx - output Node pointer
; Returns:
;   rax - 0 on success, -1 on error
activation_sigmoid_forward:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8
    
    mov rbx, rdx                ; save output pointer
    mov rdi, rsi                ; input node
    call node_sigmoid
    
    test rax, rax
    jz .sigmoid_fwd_error
    
    mov [rbx], rax              ; store output node
    xor eax, eax                ; return 0 = success
    jmp .sigmoid_fwd_done
    
.sigmoid_fwd_error:
    mov eax, -1
    
.sigmoid_fwd_done:
    add rsp, 8
    pop rbx
    pop rbp
    ret

; activation_tanh_forward - Tanh forward for loaded model
; Arguments:
;   rdi - Module pointer (unused)
;   rsi - input Node
;   rdx - output Node pointer
; Returns:
;   rax - 0 on success, -1 on error
activation_tanh_forward:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8
    
    mov rbx, rdx                ; save output pointer
    mov rdi, rsi                ; input node
    call node_tanh
    
    test rax, rax
    jz .tanh_fwd_error
    
    mov [rbx], rax              ; store output node
    xor eax, eax                ; return 0 = success
    jmp .tanh_fwd_done
    
.tanh_fwd_error:
    mov eax, -1
    
.tanh_fwd_done:
    add rsp, 8
    pop rbx
    pop rbp
    ret

; activation_softmax_forward - Softmax forward for loaded model
; Arguments:
;   rdi - Module pointer (unused)
;   rsi - input Node
;   rdx - output Node pointer
; Returns:
;   rax - 0 on success, -1 on error
activation_softmax_forward:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8
    
    mov rbx, rdx                ; save output pointer
    mov rdi, rsi                ; input node
    call node_softmax
    
    test rax, rax
    jz .softmax_fwd_error
    
    mov [rbx], rax              ; store output node
    xor eax, eax                ; return 0 = success
    jmp .softmax_fwd_done
    
.softmax_fwd_error:
    mov eax, -1
    
.softmax_fwd_done:
    add rsp, 8
    pop rbx
    pop rbp
    ret

; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
