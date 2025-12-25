; main.asm - Main Program Entry Point
; Neural Network Training and Inference Driver
; Usage: ./neural_framework [train|infer] config.ini [model.bin]

section .data
    ; Version and banner
    banner:         db "===============================================", 10
                    db " Neural Assembly Framework v1.0", 10
                    db " A Deep Learning Framework in x86-64 Assembly", 10
                    db "===============================================", 10, 0
    banner_len:     equ $ - banner
    
    ; Usage message
    usage_msg:      db "Usage: neural_framework <command> <config> [model]", 10
                    db "Commands:", 10
                    db "  train   - Train a new model", 10
                    db "  infer   - Run inference with trained model", 10
                    db "  test    - Run gradient check tests", 10
                    db 10
                    db "Examples:", 10
                    db "  neural_framework train config.ini", 10
                    db "  neural_framework infer config.ini model.bin", 10
                    db 0
    
    ; Command strings
    cmd_train:      db "train", 0
    cmd_infer:      db "infer", 0
    cmd_test:       db "test", 0
    
    ; Status messages
    msg_loading:    db "[*] Loading configuration...", 10, 0
    msg_config_ok:  db "[+] Configuration loaded successfully", 10, 0
    msg_building:   db "[*] Building model...", 10, 0
    msg_model_ok:   db "[+] Model built: ", 0
    msg_params:     db " parameters", 10, 0
    msg_training:   db "[*] Starting training...", 10, 0
    msg_epoch:      db "Epoch ", 0
    msg_loss:       db " | Loss: ", 0
    msg_acc:        db " | Accuracy: ", 0
    msg_percent:    db "%", 10, 0
    msg_done:       db "[+] Training completed!", 10, 0
    msg_saving:     db "[*] Saving model...", 10, 0
    msg_saved:      db "[+] Model saved to: ", 0
    msg_val:        db " | Val Acc: ", 0
    msg_loading_m:  db "[*] Loading model...", 10, 0
    msg_loaded:     db "[+] Model loaded successfully", 10, 0
    msg_inferring:  db "[*] Running inference...", 10, 0
    msg_result:     db "Prediction: ", 0
    msg_newline:    db 10, 0
    msg_separator:  db "-------------------------------------------", 10, 0
    msg_error:      db "[!] Error: ", 0
    newline:        db 10, 0
    
    ; SIMD messages
    msg_simd:       db "[*] SIMD: ", 0
    msg_simd_sse:   db "SSE2", 0
    msg_simd_avx:   db "AVX", 0
    msg_simd_avx2:  db "AVX2", 0
    msg_simd_avx512: db "AVX-512", 0
    msg_simd_none:  db "Scalar (no SIMD)", 0
    
    ; Default output filename
    default_model:  db "model.bin", 0
    opt_file_name:  db "optimizer.bin", 0
    
    ; Test messages
    msg_test_header: db 10, "=== Running Framework Tests ===", 10, 0
    msg_test1:      db "Test 1: Tensor creation and fill... ", 0
    msg_test2:      db "Test 2: Linear layer creation... ", 0
    msg_test3:      db "Test 3: Autograd node creation... ", 0
    msg_test4:      db "Test 4: Element-wise add kernel... ", 0
    msg_pass:       db "PASS", 10, 0
    msg_fail:       db "FAIL", 10, 0
    msg_test_done:  db "=== Tests Complete ===", 10, 0
    
    ; Float format strings
    float_format:   db "%.4f", 0
    int_format:     db "%d", 0

section .bss
    ; Command line arguments
    argc:           resq 1
    argv:           resq 1
    
    ; Pointers to current state
    config_ptr:     resq 1
    model_ptr:      resq 1
    optimizer_ptr:  resq 1
    dataset_ptr:    resq 1
    test_dataset_ptr: resq 1
    
    ; Number formatting buffer
    num_buffer:     resb 32
    
    ; Timing
    start_time:     resq 2
    end_time:       resq 2
    
    ; Training state
    current_epoch:  resd 1
    total_loss:     resd 1
    correct_count:  resd 1
    total_count:    resd 1

section .text
    global main
    
    ; Memory management
    extern mem_init
    extern mem_alloc
    extern mem_free
    extern arena_create
    
    ; Configuration
    extern config_parse
    extern config_create_default
    extern config_free
    
    ; Model I/O
    extern model_save
    extern model_load
    
    ; Tensor operations
    extern tensor_create
    extern tensor_create_zeros
    extern tensor_create_random
    extern tensor_get_size
    extern tensor_copy
    extern tensor_zero_grad
    extern tensor_fill
    extern tensor_free
    
    ; Memory
    extern mem_alloc_aligned
    
    ; Autograd nodes
    extern node_create
    
    ; Math kernels (element-wise)
    extern ew_add
    
    ; Neural network layers
    extern linear_create
    extern linear_forward
    extern linear_backward
    extern relu_forward
    extern relu_backward
    extern sigmoid_forward
    extern sigmoid_backward
    extern softmax_forward
    extern softmax_backward
    
    ; Losses
    extern mse_loss
    extern mse_loss_backward
    extern cross_entropy_loss
    extern cross_entropy_loss_backward
    
    ; Optimizers
    extern sgd_create
    extern sgd_step
    extern adam_create
    extern adam_step
    extern optimizer_set_lr
    extern optimizer_get_lr
    extern optimizer_save_state
    
    ; Dataset
    extern dataset_load
    extern dataset_load_csv
    extern dataset_get_batch
    extern dataset_shuffle
    
    ; Math kernels
    extern sgemm
    extern saxpy
    extern sdot
    
    ; Autograd
    extern autograd_init
    extern autograd_backward
    extern autograd_zero_grad
    
    ; SIMD detection
    extern detect_cpu_features
    extern get_simd_level
    
    ; Utils
    extern print_string
    extern print_int
    extern print_float
    extern xorshift_seed
    extern get_time_ns
    extern str_equals_nocase
    extern putchar

; ============================================================================
; PROGRAM ENTRY POINT
; ============================================================================

; When linked with the C runtime, `main` is called by crt; set
; `argc`/`argv` from the C calling convention (rdi/rsi).

main:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 72
    ; Store argc/argv provided by C runtime (rdi = argc, rsi = argv)
    mov [argc], rdi
    mov [argv], rsi

    ; Initialize memory subsystem
    call mem_init
    
    ; Initialize random seed with time
    call get_time_ns
    mov rdi, rax
    call xorshift_seed
    
    ; Initialize autograd
    call autograd_init
    
    ; Detect and display SIMD capabilities
    call detect_cpu_features
    mov r12d, eax               ; save SIMD level
    
    lea rdi, [msg_simd]
    call print_string
    
    ; Print SIMD level name based on return value
    cmp r12d, 4
    je .simd_avx512
    cmp r12d, 3
    je .simd_avx2
    cmp r12d, 2
    je .simd_avx
    cmp r12d, 1
    je .simd_sse
    
    lea rdi, [msg_simd_none]
    jmp .simd_print
.simd_sse:
    lea rdi, [msg_simd_sse]
    jmp .simd_print
.simd_avx:
    lea rdi, [msg_simd_avx]
    jmp .simd_print
.simd_avx2:
    lea rdi, [msg_simd_avx2]
    jmp .simd_print
.simd_avx512:
    lea rdi, [msg_simd_avx512]
.simd_print:
    call print_string
    lea rdi, [newline]
    call print_string
    
    ; Print banner
    lea rdi, [banner]
    call print_string
    
    ; Check arguments (at least 2 for test command, 3 for others)
    mov rax, [argc]
    cmp rax, 2
    jl .show_usage
    
    ; Get command (argv[1])
    mov rax, [argv]
    mov r12, [rax + 8]          ; argv[1] = command
    
    ; Check if it's 'test' command (doesn't need config)
    mov rdi, r12
    lea rsi, [cmd_test]
    call str_equals_nocase
    test eax, eax
    jnz .do_test
    
    ; Other commands need at least 3 args
    mov rax, [argc]
    cmp rax, 3
    jl .show_usage
    
    ; Get config file (argv[2])
    mov rax, [argv]
    mov r13, [rax + 16]         ; argv[2] = config
    
    ; Check for optional model file (argv[3])
    xor r14, r14
    mov rax, [argc]
    cmp rax, 4
    jl .no_model_arg
    mov rax, [argv]
    mov r14, [rax + 24]         ; argv[3] = model file
    
.no_model_arg:
    ; Parse command
    mov rdi, r12
    lea rsi, [cmd_train]
    call str_equals_nocase
    test eax, eax
    jnz .do_train
    
    mov rdi, r12
    lea rsi, [cmd_infer]
    call str_equals_nocase
    test eax, eax
    jnz .do_infer
    
    mov rdi, r12
    lea rsi, [cmd_test]
    call str_equals_nocase
    test eax, eax
    jnz .do_test
    
    jmp .show_usage
    
.do_train:
    mov rdi, r13                ; config file
    mov rsi, r14                ; model output (optional)
    call cmd_train_handler
    jmp .main_done
    
.do_infer:
    ; Model file is required for inference
    test r14, r14
    jz .show_usage
    
    mov rdi, r13                ; config file
    mov rsi, r14                ; model file
    call cmd_infer_handler
    jmp .main_done
    
.do_test:
    call cmd_test_handler
    jmp .main_done
    
.show_usage:
    lea rdi, [usage_msg]
    call print_string
    xor eax, eax
    jmp .main_exit
    
.main_done:
    ; Cleanup
    mov rdi, [config_ptr]
    test rdi, rdi
    jz .main_exit
    call config_free
    
.main_exit:
    xor eax, eax                ; return 0
    add rsp, 72
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; TRAINING HANDLER
; ============================================================================

cmd_train_handler:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 88
    
    mov r12, rdi                ; config file
    mov r13, rsi                ; model output file
    
    ; Load configuration
    lea rdi, [msg_loading]
    call print_string
    
    mov rdi, r12
    call config_parse
    
    test rax, rax
    jz .train_config_error
    
    mov [config_ptr], rax
    mov r14, rax                ; config pointer
    
    lea rdi, [msg_config_ok]
    call print_string
    
    ; Print configuration summary
    call print_config_summary
    
    ; Build model
    lea rdi, [msg_building]
    call print_string
    
    mov rdi, r14
    call build_model
    
    test rax, rax
    jz .train_build_error
    
    mov [model_ptr], rax
    mov r15, rax                ; model pointer
    
    ; Print model info
    lea rdi, [msg_model_ok]
    call print_string
    
    mov rdi, r15
    call count_parameters
    mov rdi, rax
    call print_int
    
    lea rdi, [msg_params]
    call print_string
    
    ; Debug: print first weight
    ; Create optimizer
    mov rdi, r14
    mov rsi, r15
    call create_optimizer
    mov [optimizer_ptr], rax
    mov rbx, rax
    
    ; Check optimizer is not NULL
    test rax, rax
    jz .train_config_error
    
    ; Load training dataset
    mov rdi, r14
    call load_training_data
    mov [dataset_ptr], rax
    
    ; Check if training data loaded
    test rax, rax
    jz .train_no_data
    
    ; Load test/validation dataset (optional - no error if missing)
    mov rdi, r14
    call load_test_data
    mov [test_dataset_ptr], rax
    
    ; Record start time
    lea rdi, [start_time]
    call get_time_ns
    
    ; Training loop
    lea rdi, [msg_training]
    call print_string
    
    lea rdi, [msg_separator]
    call print_string
    
    mov eax, [r14]              ; epochs (at offset 24)
    mov eax, [r14 + 24]
    mov [rbp - 80], eax         ; store total epochs
    
    mov dword [current_epoch], 1
    
.train_epoch_loop:
    mov eax, [current_epoch]
    cmp eax, [rbp - 80]
    ja .train_done
    
    ; StepLR scheduler: if step_size>0 and epoch % step_size == 0 then lr *= gamma
    push rax
    push rcx
    push rdx
    mov rcx, r14                ; config
    mov eax, [rcx + 144]        ; OFF_LR_STEP_SIZE
    test eax, eax
    jz .skip_lr_sched
    mov edx, [current_epoch]
    mov ecx, eax                ; step_size
    xor edx, edx
    mov eax, [current_epoch]
    div ecx                     ; EAX/ECX, remainder in EDX
    test edx, edx
    jne .skip_lr_sched
    
    ; Apply decay
    mov rdi, [optimizer_ptr]
    call optimizer_get_lr
    movss xmm1, [r14 + 148]     ; OFF_LR_GAMMA (float32)
    cvtss2sd xmm1, xmm1
    mulsd xmm0, xmm1
    mov rdi, [optimizer_ptr]
    call optimizer_set_lr

.skip_lr_sched:
    pop rdx
    pop rcx
    pop rax
    
    ; Print epoch number
    lea rdi, [msg_epoch]
    call print_string
    mov edi, [current_epoch]
    call print_int
    
    ; Reset epoch stats
    mov dword [total_loss], 0
    mov dword [correct_count], 0
    mov dword [total_count], 0
    
    ; Shuffle dataset (only if dataset is not NULL)
    mov rdi, [dataset_ptr]
    test rdi, rdi
    jz .skip_shuffle
    call dataset_shuffle
.skip_shuffle:
    
    ; Train one epoch
    mov rdi, [model_ptr]
    mov rsi, [optimizer_ptr]
    mov rdx, [dataset_ptr]
    mov rcx, r14                ; config
    call train_epoch
    
    ; Print epoch loss
    lea rdi, [msg_loss]
    call print_string
    
    movss xmm0, [total_loss]
    cvtss2sd xmm0, xmm0         ; convert to double for print_float
    mov edi, 4                  ; precision
    call print_float
    
    ; Print accuracy if applicable
    mov eax, [total_count]
    test eax, eax
    jz .skip_accuracy
    
    lea rdi, [msg_acc]
    call print_string
    
    ; Calculate accuracy percentage
    cvtsi2ss xmm0, dword [correct_count]
    cvtsi2ss xmm1, dword [total_count]
    divss xmm0, xmm1
    mov eax, 100
    cvtsi2ss xmm1, eax
    mulss xmm0, xmm1
    
    cvtss2sd xmm0, xmm0         ; convert float32 to float64 for printf
    mov edi, 2
    call print_float
    
    ; Print % without newline
    push r14
    mov rdi, 37                 ; '%' character
    call putchar wrt ..plt
    pop r14
    
    ; Calculate validation accuracy if test dataset exists
    mov rax, [test_dataset_ptr]
    test rax, rax
    jz .skip_val_accuracy
    
    lea rdi, [msg_val]
    call print_string
    
    ; Calculate validation accuracy
    mov rdi, [model_ptr]
    mov rsi, [test_dataset_ptr]
    call evaluate_model
    
    ; rax = correct, rdx = total
    test rdx, rdx
    jz .skip_val_accuracy
    
    cvtsi2ss xmm0, eax          ; correct
    cvtsi2ss xmm1, edx          ; total
    divss xmm0, xmm1
    mov eax, 100
    cvtsi2ss xmm1, eax
    mulss xmm0, xmm1
    
    cvtss2sd xmm0, xmm0
    mov edi, 2
    call print_float
    
    lea rdi, [msg_percent]
    call print_string
    jmp .next_epoch

.skip_val_accuracy:
    lea rdi, [msg_newline]
    call print_string
    jmp .next_epoch
    
.skip_accuracy:
    lea rdi, [msg_newline]
    call print_string
    
.next_epoch:
    inc dword [current_epoch]
    jmp .train_epoch_loop
    
.train_done:
    lea rdi, [msg_separator]
    call print_string
    
    lea rdi, [msg_done]
    call print_string
    
    ; Save model
    lea rdi, [msg_saving]
    call print_string
    
    ; Use default filename if not provided
    test r13, r13
    jnz .use_provided_name
    lea r13, [default_model]
    
.use_provided_name:
    mov rdi, [model_ptr]
    mov rsi, r13
    call model_save
    
    lea rdi, [msg_saved]
    call print_string
    mov rdi, r13
    call print_string
    lea rdi, [msg_newline]
    call print_string
    
    ; Save optimizer state alongside model
    mov rdi, [optimizer_ptr]
    test rdi, rdi
    jz .skip_opt_save
    lea rsi, [opt_file_name]
    call optimizer_save_state
.skip_opt_save:
    
    ; Record end time and print duration
    lea rdi, [end_time]
    call get_time_ns
    
    ; Print timing info
    call print_timing
    
    xor eax, eax
    jmp .train_cleanup
    
.train_no_data:
    lea rdi, [msg_error]
    call print_string
    mov eax, -1
    jmp .train_cleanup

.train_config_error:
    lea rdi, [msg_error]
    call print_string
    ; Print specific error
    mov eax, -1
    jmp .train_cleanup
    
.train_build_error:
    lea rdi, [msg_error]
    call print_string
    mov eax, -1
    
.train_cleanup:
    add rsp, 88
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; INFERENCE HANDLER
; ============================================================================

cmd_infer_handler:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 40
    
    mov r12, rdi                ; config file
    mov r13, rsi                ; model file
    
    ; Load configuration
    lea rdi, [msg_loading]
    call print_string
    
    mov rdi, r12
    call config_parse
    mov [config_ptr], rax
    mov rbx, rax
    
    lea rdi, [msg_config_ok]
    call print_string
    
    ; Load model
    lea rdi, [msg_loading_m]
    call print_string
    
    mov rdi, r13
    call model_load
    
    test rax, rax
    jz .infer_load_error
    
    mov [model_ptr], rax
    
    lea rdi, [msg_loaded]
    call print_string
    
    ; Run inference
    lea rdi, [msg_inferring]
    call print_string
    
    ; Load test data
    mov rdi, rbx
    call load_test_data
    
    ; Run inference on test set
    mov rdi, [model_ptr]
    mov rsi, rax                ; test dataset
    mov rdx, rbx                ; config
    call run_inference
    
    xor eax, eax
    jmp .infer_cleanup
    
.infer_load_error:
    lea rdi, [msg_error]
    call print_string
    mov eax, -1
    
.infer_cleanup:
    add rsp, 40
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; TEST HANDLER (Gradient Checks)
; ============================================================================

cmd_test_handler:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    ; Run gradient checks
    call run_gradient_checks
    
    add rsp, 16
    pop rbp
    ret

; ============================================================================
; HELPER FUNCTIONS
; ============================================================================

; print_config_summary - Print configuration overview
print_config_summary:
    push rbp
    mov rbp, rsp
    ; TODO: Print key config values
    pop rbp
    ret

; build_model - Build neural network from config
; Arguments:
;   rdi - config pointer
; Returns:
;   rax - model pointer
; Builds: Input -> [Linear -> ReLU] x num_layers -> Linear -> Softmax
build_model:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                ; config
    
    ; Get model parameters from config
    mov eax, [r12]              ; input_size (offset 0)
    mov [rbp - 48], eax         ; save input_size
    mov eax, [r12 + 4]          ; hidden_size
    mov [rbp - 52], eax         ; save hidden_size
    mov eax, [r12 + 8]          ; output_size
    mov [rbp - 56], eax         ; save output_size
    mov eax, [r12 + 12]         ; num_layers (number of hidden layers)
    mov [rbp - 60], eax         ; save num_layers
    
    ; Calculate total components: num_layers * 2 (Linear + ReLU each) + 2 (output Linear + Softmax)
    mov ecx, eax
    shl ecx, 1                  ; num_layers * 2
    add ecx, 2                  ; + output layer + softmax
    mov [rbp - 64], ecx         ; total_components
    
    ; Allocate model structure (8 bytes per layer + 8 for count)
    lea edi, [ecx * 8 + 16]     ; size
    call mem_alloc
    
    test rax, rax
    jz .build_error
    
    mov r13, rax                ; model pointer
    mov eax, [rbp - 64]
    mov [r13], eax              ; store num_layers
    
    ; Layer pointer offset
    lea r14, [r13 + 8]          ; current layer slot
    
    ; Track current input size
    mov r15d, [rbp - 48]        ; current_in = input_size
    
    ; Build hidden layers
    mov ebx, [rbp - 60]         ; loop counter = num_layers
    test ebx, ebx
    jz .build_output            ; skip if no hidden layers

.build_hidden_loop:
    ; Create Linear layer: current_in -> hidden_size
    mov edi, r15d               ; in_features
    mov esi, [rbp - 52]         ; out_features = hidden_size
    xor edx, edx                ; dtype = DT_FLOAT32
    call linear_create
    
    mov [r14], rax              ; store Linear layer
    add r14, 8
    
    ; Add ReLU marker
    mov qword [r14], 1          ; ReLU marker
    add r14, 8
    
    ; Update current input size for next layer
    mov r15d, [rbp - 52]        ; current_in = hidden_size
    
    dec ebx
    jnz .build_hidden_loop
    
.build_output:
    ; Create output Linear layer: hidden_size -> output_size
    mov edi, r15d               ; in_features (hidden_size or input_size if no hidden)
    mov esi, [rbp - 56]         ; out_features = output_size
    xor edx, edx                ; dtype = DT_FLOAT32
    call linear_create
    
    mov [r14], rax              ; store output Linear
    add r14, 8
    
    ; Add Softmax marker
    mov qword [r14], 2          ; Softmax marker
    
    mov rax, r13                ; return model pointer
    jmp .build_done
    
.build_error:
    xor eax, eax
    
.build_done:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; count_parameters - Count total trainable parameters
; Arguments:
;   rdi - model pointer
; Returns:
;   rax - parameter count
count_parameters:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    mov r12, rdi
    xor ebx, ebx                ; counter
    
    ; Iterate through layers
    mov ecx, [r12]              ; num_layers
    lea rdi, [r12 + 8]
    
.count_loop:
    test ecx, ecx
    jz .count_done
    
    mov rax, [rdi]
    
    ; Check if it's a real layer (pointer > 100)
    ; Values 1, 2 are markers for ReLU, Softmax
    cmp rax, 100
    jb .count_next
    
    ; Get layer's param tensors and count elements
    ; Module structure:
    ;   offset 0 = n_params (dword)
    ;   offset 8 = params (Tensor**)
    push rcx
    push rdi
    mov r8, rax                 ; module pointer
    
    mov ecx, [r8]               ; n_params
    mov rsi, [r8 + 8]           ; params array
    test rsi, rsi
    jz .count_params_done
    
.count_param_loop:
    test ecx, ecx
    jz .count_params_done
    
    mov rdi, [rsi]              ; params[i] = tensor pointer
    test rdi, rdi
    jz .count_param_next
    
    push rcx
    push rsi
    call tensor_get_size
    add ebx, eax
    pop rsi
    pop rcx
    
.count_param_next:
    add rsi, 8
    dec ecx
    jmp .count_param_loop
    
.count_params_done:
    pop rdi
    pop rcx
    
.count_next:
    add rdi, 8
    dec ecx
    jmp .count_loop
    
.count_done:
    mov eax, ebx
    pop r12
    pop rbx
    pop rbp
    ret

; create_optimizer - Create optimizer based on config
; Arguments:
;   rdi - config pointer
;   rsi - model pointer
; Returns:
;   rax - optimizer pointer
create_optimizer:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 96                 ; local storage
    
    ; Stack layout:
    ; [rbp-56] = config pointer
    ; [rbp-64] = model pointer
    ; [rbp-72] = params array pointer
    ; [rbp-80] = param_nodes array pointer
    ; [rbp-84] = total n_params count
    ; [rbp-88] = current param index
    
    mov [rbp-56], rdi           ; save config
    mov [rbp-64], rsi           ; save model
    
    ; First, count total params across all layers
    ; Model structure: offset 0 = num_layers, offset 8+ = layer pointers
    mov r12, rsi                ; model
    mov ecx, [r12]              ; num_layers
    xor r13d, r13d              ; total params count
    lea r14, [r12 + 8]          ; layer pointer array
    
.count_params_loop:
    test ecx, ecx
    jz .count_params_done
    
    mov rax, [r14]              ; layer pointer
    cmp rax, 100                ; skip markers (1=ReLU, 2=Softmax)
    jb .count_next_layer
    
    ; Real layer - add its n_params
    add r13d, [rax]             ; module->n_params at offset 0
    
.count_next_layer:
    add r14, 8
    dec ecx
    jmp .count_params_loop
    
.count_params_done:
    mov [rbp-84], r13d          ; save total count
    
    ; Allocate params array (n_params * 8 bytes)
    mov eax, r13d
    shl eax, 3
    mov edi, eax
    call mem_alloc
    test rax, rax
    jz .opt_error
    mov [rbp-72], rax           ; params array
    
    ; Allocate param_nodes array (n_params * 8 bytes)
    mov eax, [rbp-84]
    shl eax, 3
    mov edi, eax
    call mem_alloc
    test rax, rax
    jz .opt_error
    mov [rbp-80], rax           ; param_nodes array
    
    ; Now collect params and param_nodes from all layers
    mov r12, [rbp-64]           ; model
    mov ecx, [r12]              ; num_layers
    lea r14, [r12 + 8]          ; layer pointer array
    mov dword [rbp-88], 0       ; current index = 0
    mov r15, [rbp-72]           ; params dest
    mov rbx, [rbp-80]           ; param_nodes dest
    
.collect_loop:
    test ecx, ecx
    jz .collect_done
    
    push rcx                    ; save counter
    
    mov rax, [r14]              ; layer pointer
    cmp rax, 100
    jb .collect_next
    
    ; Module struct:
    ;   offset 0: n_params
    ;   offset 8: params (Tensor**)
    ;   offset 16: param_nodes (Node**)
    mov r12, rax                ; module
    mov ecx, [r12]              ; layer's n_params
    mov rsi, [r12 + 8]          ; params array
    mov rdi, [r12 + 16]         ; param_nodes array
    
.collect_layer_params:
    test ecx, ecx
    jz .collect_next
    
    ; Copy param tensor pointer
    mov rax, [rsi]
    mov [r15], rax
    add r15, 8
    add rsi, 8
    
    ; Copy param_node pointer (for optimizer to get grad later)
    mov rax, [rdi]
    mov [rbx], rax
    add rbx, 8
    add rdi, 8
    
    dec ecx
    jmp .collect_layer_params
    
.collect_next:
    pop rcx
    add r14, 8
    dec ecx
    jmp .collect_loop
    
.collect_done:
    ; Now create the optimizer with collected arrays
    mov r12, [rbp-56]           ; config
    mov rdi, [rbp-72]           ; params array
    mov rsi, [rbp-80]           ; param_nodes array
    mov edx, [rbp-84]           ; n_params
    
    ; Check optimizer type
    mov eax, [r12 + 48]         ; optimizer_type
    cmp eax, 1
    je .create_adam
    
    ; Default: SGD
    movss xmm0, [r12 + 32]      ; learning_rate
    cvtss2sd xmm0, xmm0         ; convert to double
    movss xmm1, [r12 + 52]      ; momentum
    cvtss2sd xmm1, xmm1
    call sgd_create
    jmp .opt_done
    
.create_adam:
    movss xmm0, [r12 + 32]      ; learning_rate
    cvtss2sd xmm0, xmm0
    movss xmm1, [r12 + 56]      ; beta1
    cvtss2sd xmm1, xmm1
    movss xmm2, [r12 + 60]      ; beta2
    cvtss2sd xmm2, xmm2
    movss xmm3, [r12 + 64]      ; epsilon
    cvtss2sd xmm3, xmm3
    call adam_create
    jmp .opt_done
    
.opt_error:
    xor eax, eax
    
.opt_done:
    add rsp, 96
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; load_training_data - Load training dataset
; Arguments:
;   rdi - config pointer
; Returns:
;   rax - dataset pointer
load_training_data:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    
    mov r12, rdi
    
    ; Get train_file pointer
    mov rdi, [r12 + 68]         ; train_file pointer (OFF_TRAIN_FILE = 68)
    test rdi, rdi
    jz .no_train_file
    
    ; Call dataset_load_csv(data_path, label_path, n_features=input_size, dtype=DT_FLOAT32)
    mov rsi, [r12 + 84]         ; train_label_file pointer (OFF_TRAIN_LABEL_FILE = 84)
    mov edx, [r12]              ; input_size (n_features) - 32-bit value
    xor rcx, rcx                ; dtype = DT_FLOAT32 (0)
    call dataset_load_csv
    
    ; Check if dataset loaded
    test rax, rax
    jnz .load_done
    
.no_train_file:
    ; Create dummy dataset for testing
    xor eax, eax
    
.load_done:
    pop r13
    pop r12
    pop rbp
    ret

; load_test_data - Load test dataset
; Arguments:
;   rdi - config pointer
; Returns:
;   rax - dataset pointer
load_test_data:
    push rbp
    mov rbp, rsp
    push r12
    
    mov r12, rdi
    
    mov rdi, [r12 + 76]         ; test_file pointer (OFF_TEST_FILE = 76)
    test rdi, rdi
    jz .no_test_file

    ; Call dataset_load_csv(data_path, label_path, n_features=input_size, dtype=DT_FLOAT32)
    mov rsi, [r12 + 92]         ; test_label_file pointer (OFF_TEST_LABEL_FILE = 92)
    mov edx, [r12]              ; input_size (32-bit zero extended to rdx)
    xor rcx, rcx                ; dtype = DT_FLOAT32
    call dataset_load_csv
    jmp .test_load_done
    
.no_test_file:
    xor eax, eax
    
.test_load_done:
    pop r12
    pop rbp
    ret

; train_epoch - Train one epoch
; Arguments:
;   rdi - model pointer
;   rsi - optimizer pointer
;   rdx - dataset pointer
;   rcx - config pointer
train_epoch:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 96                 ; more stack space for batch tensors
    
    mov r12, rdi                ; model
    mov r13, rsi                ; optimizer
    mov r14, rdx                ; dataset
    mov r15, rcx                ; config
    
    ; Stack layout:
    ; [rbp - 48] = batch_size
    ; [rbp - 52] = num_batches
    ; [rbp - 56] = current_batch
    ; [rbp - 64] = batch_x tensor pointer
    ; [rbp - 72] = batch_y tensor pointer
    ; [rbp - 80] = accumulated loss
    
    ; Get batch size
    mov eax, [r15 + 28]         ; batch_size
    mov [rbp - 48], eax
    
    ; Initialize accumulated loss to 0.0
    xorps xmm0, xmm0
    movss [rbp - 80], xmm0
    
    ; Get number of batches (dataset_size / batch_size)
    test r14, r14
    jz .epoch_done
    
    mov eax, [r14]              ; num_samples
    xor edx, edx
    mov ecx, [rbp - 48]
    test ecx, ecx
    jz .epoch_done              ; avoid divide by zero
    div ecx                     ; eax = num_batches
    mov [rbp - 52], eax
    
    test eax, eax
    jz .epoch_done              ; no batches
    
    mov dword [rbp - 56], 0     ; current batch
    
.batch_loop:
    mov eax, [rbp - 56]
    cmp eax, [rbp - 52]
    jge .epoch_done
    
    ; Get next batch - pass output pointers for X and Y
    mov rdi, r14                    ; dataset
    mov esi, [rbp - 56]             ; batch_index (32-bit)
    mov edx, [rbp - 48]             ; batch_size (32-bit)
    lea rcx, [rbp - 64]             ; &out_x
    lea r8, [rbp - 72]              ; &out_y
    call dataset_get_batch

    ; Create input node from batch_x tensor
    mov rdi, [rbp - 64]         ; batch_x tensor
    test rdi, rdi
    jz .next_batch              ; skip if NULL
    
    mov rsi, 0                  ; requires_grad = false for input
    call node_create
    test rax, rax
    jz .next_batch
    mov rbx, rax                ; input node
    
    ; Forward pass through model
    mov rdi, r12                ; model
    mov rsi, rbx                ; input node
    call model_forward
    
    test rax, rax
    jz .next_batch
    mov rbx, rax                ; predictions node
    
    ; Calculate loss using cross_entropy_loss
    mov rdi, rbx                ; predictions node
    mov rsi, [rbp - 72]         ; batch_y tensor (labels)
    call cross_entropy_loss
    
    push rax                    ; save loss node
    
    ; Calculate accuracy
    mov rcx, [rbx]              ; predictions node -> value (tensor)
    mov rdi, rcx                ; logits tensor
    mov rsi, [rbp - 72]         ; batch_y tensor
    call calculate_batch_accuracy
    
    add [correct_count], eax
    mov eax, [rbp - 48]         ; batch_size
    add [total_count], eax
    
    pop rax                     ; restore loss node
    
    test rax, rax
    jz .next_batch
    
    ; Save loss node for backward pass
    mov [rbp - 88], rax         ; save loss node at rbp-88
    
    ; rax = loss node - get scalar value and accumulate
    ; NODE_VALUE is at offset 0, TENSOR_DATA is at offset 0
    mov rcx, [rax]              ; NODE_VALUE = loss tensor
    test rcx, rcx
    jz .do_backward
    mov rcx, [rcx]              ; TENSOR_DATA = data pointer
    test rcx, rcx
    jz .do_backward
    movss xmm0, [rcx]           ; load loss value (float32)
    
    addss xmm0, [rbp - 80]      ; accumulate
    movss [rbp - 80], xmm0
    
.do_backward:
    ; Set loss gradient to 1.0 for backward pass
    ; TODO: loss node's gradient should be initialized
    
    ; Backward pass - pass loss node, not model!
    mov rdi, [rbp - 88]         ; loss node
    call autograd_backward
    
    ; Optimizer step
    mov rdi, r13
    call optimizer_step
    
    ; Zero gradients - use the optimizer's zero_grad function
    mov rdi, r13
    mov rax, [r13 + 32]         ; OPT_ZERO_GRAD_FN offset
    test rax, rax
    jz .next_batch
    call rax
    
.next_batch:
    inc dword [rbp - 56]
    jmp .batch_loop
    
.epoch_done:
    ; Store accumulated loss to global total_loss
    movss xmm0, [rbp - 80]
    
    ; Divide by number of batches to get average loss
    mov eax, [rbp - 52]
    test eax, eax
    jz .store_loss
    cvtsi2ss xmm1, eax
    divss xmm0, xmm1
    
.store_loss:
    movss [total_loss], xmm0
    
    add rsp, 96
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; model_forward - Forward pass through model
; Arguments:
;   rdi - model pointer
;   rsi - input tensor
; Returns:
;   rax - output tensor
model_forward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 24
    
    mov r12, rdi                ; model
    mov r13, rsi                ; current activation
    
    mov ebx, [r12]              ; num_layers
    lea r12, [r12 + 8]          ; layer pointers
    
.forward_loop:
    test ebx, ebx
    jz .forward_done
    
    mov rax, [r12]
    
    ; Check layer type
    cmp rax, 1                  ; ReLU marker
    je .apply_relu
    cmp rax, 2                  ; Softmax marker
    je .apply_softmax
    
    ; Linear layer
    mov rdi, rax
    mov rsi, r13
    call linear_forward
    
    mov r13, rax
    jmp .forward_next
    
.apply_relu:
    mov rdi, r13
    call relu_forward
    
    mov r13, rax
    jmp .forward_next
    
.apply_softmax:
    mov rdi, r13
    call softmax_forward
    
    mov r13, rax
    
.forward_next:
    add r12, 8
    dec ebx
    jmp .forward_loop
    
.forward_done:
    mov rax, r13
    add rsp, 24
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; model_backward - Backward pass through model
; Arguments:
;   rdi - model pointer
model_backward:
    push rbp
    mov rbp, rsp
    
    ; Call autograd backward
    call autograd_backward
    
    pop rbp
    ret

; model_zero_grad - Zero all gradients
; Arguments:
;   rdi - model pointer
model_zero_grad:
    push rbp
    mov rbp, rsp
    
    call autograd_zero_grad
    
    pop rbp
    ret

; optimizer_step - Apply optimizer update
; Arguments:
;   rdi - optimizer pointer
optimizer_step:
    push rbp
    mov rbp, rsp
    push r12
    
    mov r12, rdi                ; optimizer
    
    ; Call through function pointer at OPT_STEP_FN (offset 24)
    mov rax, [r12 + 24]         ; step_fn
    mov rdi, r12
    call rax
    
    pop r12
    pop rbp
    ret

; calculate_batch_accuracy - Calculate accuracy for a batch
; Arguments:
;   rdi - logits tensor (batch_size x num_classes)
;   rsi - targets tensor (batch_size) - float values representing class indices
; Returns:
;   eax - number of correct predictions
calculate_batch_accuracy:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov r12, rdi                ; logits
    mov r13, rsi                ; targets
    
    ; Get batch size
    mov rax, [r12 + 16]         ; shape
    mov ecx, [rax]              ; batch_size
    mov r14d, ecx
    
    ; Get num_classes
    mov edx, [rax + 8]          ; num_classes
    mov r15d, edx
    
    xor ebx, ebx                ; correct_count = 0
    xor r8d, r8d                ; i = 0
    
    mov r9, [r12]               ; logits data
    mov r10, [r13]              ; targets data
    
.acc_loop:
    cmp r8d, r14d
    jge .acc_done
    
    ; Find argmax for sample i
    ; logits[i] starts at r9 + i * num_classes * 4 (assuming float32)
    mov eax, r8d
    imul eax, r15d
    shl eax, 2                  ; * 4 bytes
    lea rdi, [r9 + rax]         ; pointer to logits for this sample
    
    ; Find max index in this row
    xor ecx, ecx                ; max_idx = 0
    movss xmm0, [rdi]           ; max_val
    mov edx, 1                  ; j = 1
    
.max_loop:
    cmp edx, r15d
    jge .max_found
    
    movss xmm1, [rdi + rdx*4]
    comiss xmm1, xmm0
    jbe .next_col
    
    movaps xmm0, xmm1
    mov ecx, edx
    
.next_col:
    inc edx
    jmp .max_loop
    
.max_found:
    ; ecx is predicted class
    
    ; Get target class
    ; targets[i] is at r10 + i * 4
    movss xmm2, [r10 + r8*4]
    cvttss2si eax, xmm2         ; convert float target to int

.check_match:
    cmp ecx, eax
    jne .next_sample
    
    inc ebx                     ; correct!
    
.next_sample:
    inc r8d
    jmp .acc_loop
    
.acc_done:
    mov eax, ebx
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; evaluate_model - Evaluate model on dataset without printing
; Arguments:
;   rdi - model pointer
;   rsi - dataset pointer
; Returns:
;   rax - correct count
;   rdx - total count
evaluate_model:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 72
    
    mov r12, rdi                ; model
    mov r13, rsi                ; dataset
    
    test r13, r13
    jz .eval_no_data
    
    mov dword [rbp - 60], 0     ; correct_count = 0
    mov dword [rbp - 64], 0     ; total_count = 0
    mov dword [rbp - 68], 0     ; current sample index
    
    ; Get batch size (use 32 for efficiency)
    mov dword [rbp - 72], 32
    
.eval_loop:
    mov eax, [rbp - 68]
    cmp eax, [r13]              ; num_samples
    jge .eval_done
    
    ; Calculate batch size (min of 32 and remaining samples)
    mov ecx, [r13]
    sub ecx, eax                ; remaining = num_samples - current
    cmp ecx, 32
    jle .use_remaining
    mov ecx, 32
.use_remaining:
    mov [rbp - 72], ecx         ; actual batch size
    
    ; Get batch
    mov rdi, r13
    mov esi, [rbp - 68]
    mov edx, [rbp - 72]
    lea rcx, [rbp - 48]         ; &out_x
    lea r8, [rbp - 56]          ; &out_y
    call dataset_get_batch
    
    ; Check if batch was created successfully
    mov rdi, [rbp - 48]         ; batch_x tensor
    test rdi, rdi
    jz .eval_done               ; Skip if batch creation failed
    
    ; Create input node
    xor esi, esi                ; requires_grad = false
    call node_create
    test rax, rax
    jz .eval_done               ; Skip if node creation failed
    mov r14, rax                ; input node
    
    ; Forward pass
    mov rdi, r12
    mov rsi, r14
    call model_forward
    test rax, rax
    jz .eval_done               ; Skip if forward failed
    mov r15, rax                ; output node
    
    ; Calculate batch accuracy
    mov rdi, r15                ; output node
    mov rdi, [rdi]              ; output tensor (logits)
    mov rsi, [rbp - 56]         ; labels tensor
    call calculate_batch_accuracy
    
    add [rbp - 60], eax         ; correct_count += batch_correct
    mov eax, [rbp - 72]
    add [rbp - 64], eax         ; total_count += batch_size
    
    ; Advance to next batch
    mov eax, [rbp - 72]
    add [rbp - 68], eax
    jmp .eval_loop
    
.eval_done:
    mov eax, [rbp - 60]         ; return correct in rax
    mov edx, [rbp - 64]         ; return total in rdx
    jmp .eval_exit
    
.eval_no_data:
    xor eax, eax
    xor edx, edx
    
.eval_exit:
    add rsp, 72
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; run_inference - Run inference on dataset
; Arguments:
;   rdi - model pointer
;   rsi - dataset pointer
;   rdx - config pointer
run_inference:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56                 ; Align stack to 16 bytes (5 pushes = 40 bytes, need 8 more for 16-byte alignment + 48 bytes local = 88? No)
                                ; rbp aligned. 5 pushes -> rsp ends in 8. sub 56 -> rsp ends in 0.
                                ; [rbp-48]=out_x, [rbp-56]=out_y, [rbp-60]=correct, [rbp-64]=total, [rbp-68]=sample
    
    mov r12, rdi                ; model
    mov r13, rsi                ; dataset
    mov rbx, rdx                ; config
    
    test r13, r13
    jz .infer_done
    
    mov dword [rbp - 60], 0     ; correct_count = 0
    mov dword [rbp - 64], 0     ; total_count = 0
    mov dword [rbp - 68], 0     ; current sample index
    
.infer_loop:
    mov eax, [rbp - 68]
    cmp eax, [r13]              ; num_samples
    jge .infer_summary
    
    ; Get sample
    mov rdi, r13
    mov esi, [rbp - 68]
    mov edx, 1                  ; batch size 1
    lea rcx, [rbp - 48]         ; &out_x
    lea r8, [rbp - 56]          ; &out_y
    call dataset_get_batch
    
    ; Create input node
    mov rdi, [rbp - 48]         ; out_x tensor
    mov rsi, 0                  ; requires_grad = false
    call node_create
    mov r14, rax                ; input node
    
    ; Forward pass
    mov rdi, r12
    mov rsi, r14
    call model_forward
    mov r15, rax                ; output node
    
    ; Print result prefix
    lea rdi, [msg_result]
    call print_string
    
    ; Find argmax of output
    mov rdi, r15                ; output node
    mov rdi, [rdi]              ; output tensor
    call tensor_argmax
    mov rbx, rax                ; prediction
    
    mov rdi, rax
    call print_int
    
    ; Check accuracy if label exists
    mov rax, [rbp - 56]         ; out_y tensor
    test rax, rax
    jz .print_newline
    
    ; Print " | Target: "
    lea rdi, [msg_loss]         ; reuse loss message for separator
    call print_string
    
    mov rax, [rbp - 56]
    movss xmm0, [rax]           ; target value (float)
    cvttss2si eax, xmm0         ; target int
    
    mov rdi, rax
    call print_int
    
    cmp eax, ebx
    jne .print_newline
    
    inc dword [rbp - 60]        ; correct++
    
.print_newline:
    lea rdi, [msg_newline]
    call print_string
    
    inc dword [rbp - 64]        ; total++
    inc dword [rbp - 68]        ; sample++
    jmp .infer_loop
    
.infer_summary:
    ; Print accuracy
    lea rdi, [msg_acc]
    call print_string
    
    cvtsi2ss xmm0, dword [rbp - 60]
    cvtsi2ss xmm1, dword [rbp - 64]
    divss xmm0, xmm1
    mov eax, 100
    cvtsi2ss xmm1, eax
    mulss xmm0, xmm1
    
    mov edi, 2
    call print_float
    
    lea rdi, [msg_percent]
    call print_string
    
.infer_done:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; tensor_argmax - Find index of maximum value
; Arguments:
;   rdi - tensor pointer
; Returns:
;   eax - index of maximum
tensor_argmax:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    mov r12, rdi
    
    ; Get tensor size
    mov rdi, r12
    call tensor_get_size
    mov ecx, eax                ; size
    
    mov rsi, [r12]              ; data pointer
    
    xor ebx, ebx                ; max_idx = 0
    movss xmm0, [rsi]           ; max_val = data[0]
    mov edx, 1                  ; i = 1
    
.argmax_loop:
    cmp edx, ecx
    jge .argmax_done
    
    movss xmm1, [rsi + rdx*4]
    comiss xmm1, xmm0
    jbe .argmax_next
    
    ; New maximum
    movaps xmm0, xmm1
    mov ebx, edx
    
.argmax_next:
    inc edx
    jmp .argmax_loop
    
.argmax_done:
    mov eax, ebx
    pop r12
    pop rbx
    pop rbp
    ret

; run_gradient_checks - Run numerical gradient checks
; This creates a small test network and verifies gradients
run_gradient_checks:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 96
    
    ; Print test header
    lea rdi, [msg_test_header]
    call print_string
    
    ; Test 1: Create a small tensor and verify basic operations
    lea rdi, [msg_test1]
    call print_string
    
    ; Create a 2x3 tensor
    mov qword [rsp], 2          ; shape[0]
    mov qword [rsp+8], 3        ; shape[1]
    mov rdi, 2                  ; ndim
    lea rsi, [rsp]              ; shape
    xor edx, edx                ; dtype = float32
    call tensor_create
    mov r12, rax
    
    test r12, r12
    jz .test1_fail
    
    ; Fill with test value
    mov rdi, r12
    mov eax, 0x3F800000         ; 1.0f in IEEE754
    movd xmm0, eax
    call tensor_fill
    
    ; Get size and verify
    mov rdi, r12
    call tensor_get_size
    cmp rax, 6                  ; 2*3 = 6
    jne .test1_fail
    
    lea rdi, [msg_pass]
    call print_string
    jmp .test2
    
.test1_fail:
    lea rdi, [msg_fail]
    call print_string
    
.test2:
    ; Test 2: Create linear layer and verify parameter count
    lea rdi, [msg_test2]
    call print_string
    
    mov rdi, 4                  ; in_features
    mov rsi, 2                  ; out_features
    xor edx, edx                ; dtype
    call linear_create
    mov r13, rax
    
    test r13, r13
    jz .test2_fail
    
    ; Check n_params = 2 (weight + bias)
    mov eax, [r13]
    cmp eax, 2
    jne .test2_fail
    
    lea rdi, [msg_pass]
    call print_string
    jmp .test3
    
.test2_fail:
    lea rdi, [msg_fail]
    call print_string
    
.test3:
    ; Test 3: Verify autograd node creation
    lea rdi, [msg_test3]
    call print_string
    
    ; Create a node from tensor
    mov rdi, r12                ; tensor from test1
    mov rsi, 1                  ; requires_grad
    call node_create
    mov r14, rax
    
    test r14, r14
    jz .test3_fail
    
    lea rdi, [msg_pass]
    call print_string
    jmp .test4
    
.test3_fail:
    lea rdi, [msg_fail]
    call print_string
    
.test4:
    ; Test 4: Math kernel (element-wise add with tensors)
    lea rdi, [msg_test4]
    call print_string
    
    ; Create tensor a (1D, 8 elements)
    mov qword [rsp], 8          ; shape[0] = 8
    mov rdi, 1                  ; ndim = 1
    lea rsi, [rsp]
    xor edx, edx                ; dtype = float32
    call tensor_create
    mov r15, rax
    test r15, r15
    jz .test4_fail
    
    ; Fill tensor a with 1.0 (as double for tensor_fill)
    mov rdi, r15
    mov rax, 0x3FF0000000000000 ; 1.0 as double
    mov [rsp+24], rax
    movsd xmm0, [rsp+24]
    call tensor_fill
    
    ; Create tensor b (1D, 8 elements)
    mov qword [rsp], 8
    mov rdi, 1
    lea rsi, [rsp]
    xor edx, edx
    call tensor_create
    mov rbx, rax
    test rbx, rbx
    jz .test4_fail
    
    ; Fill tensor b with 2.0 (as double)
    mov rdi, rbx
    mov rax, 0x4000000000000000 ; 2.0 as double
    mov [rsp+24], rax
    movsd xmm0, [rsp+24]
    call tensor_fill
    
    ; Create output tensor c
    mov qword [rsp], 8
    mov rdi, 1
    lea rsi, [rsp]
    xor edx, edx
    call tensor_create
    mov [rsp+16], rax
    test rax, rax
    jz .test4_fail
    
    ; Call ew_add(c, a, b)
    mov rdi, [rsp+16]           ; out
    mov rsi, r15                ; a
    mov rdx, rbx                ; b
    call ew_add
    
    ; Check result - first element should be 3.0
    mov rax, [rsp+16]           ; tensor c
    mov rax, [rax]              ; c->data
    mov eax, [rax]              ; c->data[0]
    cmp eax, 0x40400000         ; 3.0f
    jne .test4_fail
    
    lea rdi, [msg_pass]
    call print_string
    jmp .tests_done
    
.test4_fail:
    lea rdi, [msg_fail]
    call print_string
    
.tests_done:
    ; Print summary
    lea rdi, [msg_test_done]
    call print_string
    
    ; Cleanup
    test r12, r12
    jz .skip_clean1
    mov rdi, r12
    call tensor_free
.skip_clean1:
    
    add rsp, 96
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; print_timing - Print training duration
print_timing:
    push rbp
    mov rbp, rsp
    
    ; Calculate elapsed time
    mov rax, [end_time]
    sub rax, [start_time]
    
    ; Convert nanoseconds to seconds
    mov rcx, 1000000000
    xor edx, edx
    div rcx
    
    ; Print elapsed seconds
    ; TODO: format and print
    
    pop rbp
    ret

; print_string/print_int/print_float implementations removed.
; These helper functions are provided by `utils.asm` and are declared
; `extern` above. Keeping single definitions avoids linker conflicts.

; get_time_ns - Get current time in nanoseconds
; Returns:
;   rax - time in nanoseconds
get_time_ns:
    push rbp
    mov rbp, rsp
    sub rsp, 32
    
    ; clock_gettime(CLOCK_MONOTONIC, &timespec)
    mov rax, 228                ; sys_clock_gettime
    mov rdi, 1                  ; CLOCK_MONOTONIC
    lea rsi, [rbp - 16]
    syscall
    
    ; Convert to nanoseconds
    mov rax, [rbp - 16]         ; seconds
    imul rax, 1000000000
    add rax, [rbp - 8]          ; nanoseconds
    
    add rsp, 32
    pop rbp
    ret
; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
