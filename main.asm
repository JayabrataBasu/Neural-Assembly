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
    usage_msg:      db "Usage: neural_framework <command> <config> [rel model]", 10
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
    
    ; Sequential parsing
    seq_prefix:     db "Sequential(", 0
    
    ; Activation names for parsing
    act_relu:       db "relu", 0
    act_sigmoid:    db "sigmoid", 0
    act_tanh:       db "tanh", 0
    act_softmax:    db "softmax", 0
    act_gelu:       db "gelu", 0
    act_leaky_relu: db "leaky_relu", 0
    act_elu:        db "elu", 0
    act_selu:       db "selu", 0
    act_swish:      db "swish", 0
    act_mish:       db "mish", 0
    act_hardswish:  db "hardswish", 0
    act_softplus:   db "softplus", 0
    act_hardtanh:   db "hardtanh", 0
    
    ; Layer type names for full syntax
    layer_linear:   db "Linear", 0
    layer_relu:     db "ReLU", 0
    layer_sigmoid:  db "Sigmoid", 0
    layer_tanh:     db "Tanh", 0
    layer_softmax:  db "Softmax", 0
    layer_gelu:     db "GELU", 0
    layer_leaky_relu: db "LeakyReLU", 0
    layer_elu:      db "ELU", 0
    layer_selu:     db "SELU", 0
    layer_swish:    db "Swish", 0
    layer_mish:     db "Mish", 0
    layer_hardswish: db "HardSwish", 0
    layer_softplus: db "Softplus", 0
    layer_hardtanh: db "HardTanh", 0
    layer_dropout:  db "Dropout", 0
    layer_batchnorm: db "BatchNorm", 0
    layer_flatten:  db "Flatten", 0
    
    ; Layer type constants
    LAYER_LINEAR    equ 1
    LAYER_ACTIVATION equ 2
    LAYER_DROPOUT   equ 3
    LAYER_BATCHNORM equ 4
    LAYER_FLATTEN   equ 5
    
    ; Activation type constants (must match nn_layers.asm)
    ACT_NONE        equ 0
    ACT_RELU        equ 1
    ACT_SIGMOID     equ 2
    ACT_TANH        equ 3
    ACT_SOFTMAX     equ 4
    ACT_GELU        equ 5
    ACT_LEAKY_RELU  equ 6
    ACT_ELU         equ 7
    ACT_SELU        equ 8
    ACT_SWISH       equ 9
    ACT_MISH        equ 10
    ACT_HARDSWISH   equ 11
    ACT_SOFTPLUS    equ 12
    ACT_HARDTANH    equ 13
    
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
    
    ; Config offsets
    OFF_INPUT_SIZE          equ 0
    OFF_HIDDEN_SIZE         equ 4
    OFF_OUTPUT_SIZE         equ 8
    OFF_NUM_LAYERS          equ 12
    OFF_ACTIVATION          equ 16
    OFF_DROPOUT_RATE        equ 20
    OFF_EPOCHS              equ 24
    OFF_BATCH_SIZE          equ 28
    OFF_LEARNING_RATE       equ 32
    OFF_WEIGHT_DECAY        equ 36
    OFF_EARLY_STOPPING      equ 40
    OFF_PATIENCE            equ 44
    OFF_OPTIMIZER_TYPE      equ 48
    OFF_MOMENTUM            equ 52
    OFF_BETA1               equ 56
    OFF_BETA2               equ 60
    OFF_EPSILON             equ 64
    OFF_TRAIN_FILE          equ 68
    OFF_TEST_FILE           equ 76
    OFF_TRAIN_LABEL_FILE    equ 84
    OFF_TEST_LABEL_FILE     equ 92
    OFF_VAL_SPLIT           equ 100
    OFF_SHUFFLE             equ 104
    OFF_NORMALIZE           equ 108
    OFF_HIDDEN_SIZES        equ 112
    OFF_LR_STEP_SIZE        equ 144
    OFF_LR_GAMMA            equ 148
    OFF_ARCHITECTURE        equ 152
    
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
    extern node_free
    extern node_free_graph
    
    ; Math kernels (element-wise)
    extern ew_add
    
    ; Neural network layers
    extern linear_create
    extern linear_forward
    extern module_free
    extern linear_backward
    extern relu_forward
    extern relu_backward
    extern sigmoid_forward
    extern sigmoid_backward
    extern softmax_forward
    extern softmax_backward
    extern node_sigmoid
    
    ; Sequential container
    extern neural_sequential_create
    extern neural_sequential_free
    extern neural_sequential_add
    extern neural_sequential_forward
    extern neural_sequential_size
    extern neural_sequential_get
    extern neural_sequential_parameters
    extern neural_sequential_get_intermediate
    extern neural_sequential_set_save_intermediates
    extern neural_sequential_clear_intermediates
    
    ; Activation layers
    extern activation_create
    extern activation_relu_create
    extern activation_sigmoid_create
    extern activation_tanh_create
    extern activation_softmax_create
    extern activation_gelu_create
    extern activation_leaky_relu_create
    extern activation_elu_create
    extern activation_selu_create
    extern activation_swish_create
    extern activation_mish_create
    extern activation_hardswish_create
    extern activation_softplus_create
    extern activation_hardtanh_create
    
    ; Losses
    extern mse_loss
    extern mse_loss_backward
    extern cross_entropy_loss
    extern cross_entropy_loss_backward
    extern bce_loss
    extern bce_loss_backward
    extern mse_loss
    extern mse_loss_backward
    
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
    mov [rel argc], rdi
    mov [rel argv], rsi

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
    
    lea rdi, [rel msg_simd]
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
    
    lea rdi, [rel msg_simd_none]
    jmp .simd_print
.simd_sse:
    lea rdi, [rel msg_simd_sse]
    jmp .simd_print
.simd_avx:
    lea rdi, [rel msg_simd_avx]
    jmp .simd_print
.simd_avx2:
    lea rdi, [rel msg_simd_avx2]
    jmp .simd_print
.simd_avx512:
    lea rdi, [rel msg_simd_avx512]
.simd_print:
    call print_string
    lea rdi, [rel newline]
    call print_string
    
    ; Print banner
    lea rdi, [rel banner]
    call print_string
    
    ; Check arguments (at least 2 for test command, 3 for others)
    mov rax, [rel argc]
    cmp rax, 2
    jl .show_usage
    
    ; Get command (argv[1])
    mov rax, [rel argv]
    mov r12, [rax + 8]          ; argv[1] = command
    
    ; Check if it's 'test' command (doesn't need config)
    mov rdi, r12
    lea rsi, [rel cmd_test]
    call str_equals_nocase
    test eax, eax
    jnz .do_test
    
    ; Other commands need at least 3 args
    mov rax, [rel argc]
    cmp rax, 3
    jl .show_usage
    
    ; Get config file (argv[2])
    mov rax, [rel argv]
    mov r13, [rax + 16]         ; argv[2] = config
    
    ; Check for optional model file (argv[3])
    xor r14, r14
    mov rax, [rel argc]
    cmp rax, 4
    jl .no_model_arg
    mov rax, [rel argv]
    mov r14, [rax + 24]         ; argv[3] = model file
    
.no_model_arg:
    ; Parse command
    mov rdi, r12
    lea rsi, [rel cmd_train]
    call str_equals_nocase
    test eax, eax
    jnz .do_train
    
    mov rdi, r12
    lea rsi, [rel cmd_infer]
    call str_equals_nocase
    test eax, eax
    jnz .do_infer
    
    mov rdi, r12
    lea rsi, [rel cmd_test]
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
    lea rdi, [rel usage_msg]
    call print_string
    xor eax, eax
    jmp .main_exit
    
.main_done:
    ; Cleanup
    mov rdi, [rel config_ptr]
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
    lea rdi, [rel msg_loading]
    call print_string
    
    mov rdi, r12
    call config_parse
    
    test rax, rax
    jz .train_config_error
    
    mov [rel config_ptr], rax
    mov r14, rax                ; config pointer
    
    lea rdi, [rel msg_config_ok]
    call print_string
    
    ; Print configuration summary
    call print_config_summary
    
    ; Build model
    lea rdi, [rel msg_building]
    call print_string
    
    mov rdi, r14
    call build_model
    
    test rax, rax
    jz .train_build_error
    
    mov [rel model_ptr], rax
    mov r15, rax                ; model pointer
    
    ; Print model info
    lea rdi, [rel msg_model_ok]
    call print_string
    
    mov rdi, r15
    call count_parameters
    mov rdi, rax
    call print_int
    
    lea rdi, [rel msg_params]
    call print_string
    
    ; Debug: print first weight
    ; Create optimizer
    mov rdi, r14
    mov rsi, r15
    call create_optimizer
    mov [rel optimizer_ptr], rax
    mov rbx, rax
    
    ; Check optimizer is not NULL
    test rax, rax
    jz .train_config_error
    
    ; Load training dataset
    mov rdi, r14
    call load_training_data
    mov [rel dataset_ptr], rax
    
    ; Check if training data loaded
    test rax, rax
    jz .train_no_data
    
    ; Load test/validation dataset (optional - no error if missing)
    mov rdi, r14
    call load_test_data
    mov [rel test_dataset_ptr], rax
    
    ; Record start time
    lea rdi, [rel start_time]
    call get_time_ns
    
    ; Training loop
    lea rdi, [rel msg_training]
    call print_string
    
    lea rdi, [rel msg_separator]
    call print_string
    
    mov eax, [rel r14]              ; epochs (at offset 24)
    mov eax, [r14 + 24]
    mov [rbp - 80], eax         ; store total epochs
    
    mov dword [rel current_epoch], 1
    
.train_epoch_loop:
    mov eax, [rel current_epoch]
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
    mov edx, [rel current_epoch]
    mov ecx, eax                ; step_size
    xor edx, edx
    mov eax, [rel current_epoch]
    div ecx                     ; EAX/ECX, remainder in EDX
    test edx, edx
    jne .skip_lr_sched
    
    ; Apply decay
    mov rdi, [rel optimizer_ptr]
    call optimizer_get_lr
    movss xmm1, [r14 + 148]     ; OFF_LR_GAMMA (float32)
    cvtss2sd xmm1, xmm1
    mulsd xmm0, xmm1
    mov rdi, [rel optimizer_ptr]
    call optimizer_set_lr

.skip_lr_sched:
    pop rdx
    pop rcx
    pop rax
    
    ; Print epoch number
    lea rdi, [rel msg_epoch]
    call print_string
    mov edi, [rel current_epoch]
    call print_int
    
    ; Reset epoch stats
    mov dword [rel total_loss], 0
    mov dword [rel correct_count], 0
    mov dword [rel total_count], 0
    
    ; Shuffle dataset (only if dataset is not NULL)
    mov rdi, [rel dataset_ptr]
    test rdi, rdi
    jz .skip_shuffle
    call dataset_shuffle
.skip_shuffle:
    
    ; Train one epoch
    mov rdi, [rel model_ptr]
    mov rsi, [rel optimizer_ptr]
    mov rdx, [rel dataset_ptr]
    mov rcx, r14                ; config
    call train_epoch
    
    ; Print epoch loss
    lea rdi, [rel msg_loss]
    call print_string
    
    movss xmm0, [rel total_loss]
    cvtss2sd xmm0, xmm0         ; convert to double for print_float
    mov edi, 4                  ; precision
    call print_float
    
    ; Print accuracy if applicable
    mov eax, [rel total_count]
    test eax, eax
    jz .skip_accuracy
    
    lea rdi, [rel msg_acc]
    call print_string
    
    ; Calculate accuracy percentage
    cvtsi2ss xmm0, dword [rel correct_count]
    cvtsi2ss xmm1, dword [rel total_count]
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
    mov rax, [rel test_dataset_ptr]
    test rax, rax
    jz .skip_val_accuracy
    
    lea rdi, [rel msg_val]
    call print_string
    
    ; Calculate validation accuracy
    mov rdi, [rel model_ptr]
    mov rsi, [rel test_dataset_ptr]
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
    
    lea rdi, [rel msg_percent]
    call print_string
    jmp .next_epoch

.skip_val_accuracy:
    lea rdi, [rel msg_newline]
    call print_string
    jmp .next_epoch
    
.skip_accuracy:
    lea rdi, [rel msg_newline]
    call print_string
    
.next_epoch:
    inc dword [rel current_epoch]
    jmp .train_epoch_loop
    
.train_done:
    lea rdi, [rel msg_separator]
    call print_string
    
    lea rdi, [rel msg_done]
    call print_string
    
    ; Save model
    lea rdi, [rel msg_saving]
    call print_string
    
    ; Use default filename if not provided
    test r13, r13
    jnz .use_provided_name
    lea r13, [rel default_model]
    
.use_provided_name:
    mov rdi, [rel model_ptr]
    mov rsi, r13
    call model_save
    
    lea rdi, [rel msg_saved]
    call print_string
    mov rdi, r13
    call print_string
    lea rdi, [rel msg_newline]
    call print_string
    
    ; Save optimizer state alongside model
    mov rdi, [rel optimizer_ptr]
    test rdi, rdi
    jz .skip_opt_save
    lea rsi, [rel opt_file_name]
    call optimizer_save_state
.skip_opt_save:
    
    ; Record end time and print duration
    lea rdi, [rel end_time]
    call get_time_ns
    
    ; Print timing info
    call print_timing
    
    xor eax, eax
    jmp .train_cleanup
    
.train_no_data:
    lea rdi, [rel msg_error]
    call print_string
    mov eax, -1
    jmp .train_cleanup

.train_config_error:
    lea rdi, [rel msg_error]
    call print_string
    ; Print specific error
    mov eax, -1
    jmp .train_cleanup
    
.train_build_error:
    lea rdi, [rel msg_error]
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
    lea rdi, [rel msg_loading]
    call print_string
    
    mov rdi, r12
    call config_parse
    mov [rel config_ptr], rax
    mov rbx, rax
    
    lea rdi, [rel msg_config_ok]
    call print_string
    
    ; Load model
    lea rdi, [rel msg_loading_m]
    call print_string
    
    mov rdi, r13
    call model_load
    
    test rax, rax
    jz .infer_load_error
    
    mov [rel model_ptr], rax
    
    lea rdi, [rel msg_loaded]
    call print_string
    
    ; Run inference
    lea rdi, [rel msg_inferring]
    call print_string
    
    ; Load test data
    mov rdi, rbx
    call load_test_data
    
    ; Run inference on test set
    mov rdi, [rel model_ptr]
    mov rsi, rax                ; test dataset
    mov rdx, rbx                ; config
    call run_inference
    
    xor eax, eax
    jmp .infer_cleanup
    
.infer_load_error:
    lea rdi, [rel msg_error]
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
; Or parses architecture string for Sequential models
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
    
    ; Check if architecture is specified
    mov rax, [r12 + 152]        ; OFF_ARCHITECTURE
    test rax, rax
    jz .legacy_build            ; no architecture, use legacy build
    
    ; Parse architecture string for Sequential
    mov rdi, rax
    mov rsi, r12
    call parse_architecture
    jmp .build_done
    
.legacy_build:
    ; Get model parameters from config
    mov eax, [rel r12]              ; input_size (offset 0)
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
    mov [rel r13], eax              ; store num_layers
    
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
    
    mov [rel r14], rax              ; store Linear layer
    add r14, 8
    
    ; Add ReLU marker
    mov qword [rel r14], 1          ; ReLU marker
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
    
    mov [rel r14], rax              ; store output Linear
    add r14, 8
    
    ; For binary classification (output_size <= 2), use Sigmoid; else Softmax
    mov eax, [rbp - 56]         ; output_size
    cmp eax, 2
    jg .use_softmax
    
    ; Add Sigmoid marker
    mov qword [rel r14], 3          ; Sigmoid marker (new: 3)
    jmp .build_finish
    
.use_softmax:
    ; Add Softmax marker
    mov qword [rel r14], 2          ; Softmax marker
    
.build_finish:
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
    push r13
    push r14
    
    mov r12, rdi
    xor ebx, ebx                ; counter
    
    ; Check if this is a legacy model
    mov rax, [r12 + 8]          ; check second field
    cmp rax, 0x1000             ; if < 0x1000, probably capacity (Sequential)
    jb .sequential_count        ; else, probably layer pointer (legacy)
    
    ; Legacy model format
    mov ecx, [rel r12]              ; num_layers
    lea rdi, [r12 + 8]
    
.legacy_count_loop:
    test ecx, ecx
    jz .count_done
    
    mov rax, [rel rdi]
    
    ; Check if it's a real layer (pointer > 100)
    ; Values 1, 2 are markers for ReLU, Softmax
    cmp rax, 100
    jb .legacy_count_next
    
    ; Get layer's param tensors and count elements
    push rcx
    push rdi
    mov r8, rax                 ; module pointer
    
    mov ecx, [rel r8]               ; n_params
    mov rsi, [r8 + 8]           ; params array
    test rsi, rsi
    jz .legacy_count_params_done
    
.legacy_count_param_loop:
    test ecx, ecx
    jz .legacy_count_params_done
    
    mov rdi, [rel rsi]              ; params[rel i] = tensor pointer
    test rdi, rdi
    jz .legacy_count_param_next
    
    push rcx
    push rsi
    call tensor_get_size
    add ebx, eax
    pop rsi
    pop rcx
    
.legacy_count_param_next:
    add rsi, 8
    dec ecx
    jmp .legacy_count_param_loop
    
.legacy_count_params_done:
    pop rdi
    pop rcx
    
.legacy_count_next:
    add rdi, 8
    dec ecx
    jmp .legacy_count_loop
    
    jmp .count_done
    
.sequential_count:
    ; For Sequential models, get all parameter tensors and count their sizes
    ; Save r12 (Sequential pointer) and ebx (counter, which is 0)
    ; Allocate buffer for params
    mov edi, 100*8
    call mem_alloc
    test rax, rax
    jz .count_done
    
    mov r13, rax                ; r13 = params array buffer
    
    ; Get parameters from Sequential
    mov rdi, r12                ; Sequential pointer
    mov rsi, r13                ; params array buffer
    mov rdx, 100                ; max params
    call neural_sequential_parameters
    
    test rax, rax
    jz .count_cleanup
    
    mov rcx, rax                ; rcx = param count
    mov r14, r13                ; r14 = current param pointer
    
.seq_count_loop:
    test rcx, rcx
    jz .count_cleanup
    
    mov rdi, [r14]              ; tensor pointer
    test rdi, rdi
    jz .seq_count_next
    
    push rcx
    push r14
    call tensor_get_size
    add ebx, eax                ; add to counter (ebx)
    pop r14
    pop rcx
    
.seq_count_next:
    add r14, 8                  ; next param
    dec rcx
    jmp .seq_count_loop
    
.count_cleanup:
    ; Free temporary array
    mov rdi, r13
    call mem_free
    
.count_done:
    mov eax, ebx
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; parse_architecture - Parse architecture string and build model
; Arguments:
;   rdi - architecture string
;   rsi - config pointer
; Returns:
;   rax - model pointer (Sequential or legacy)
;
; Supports three formats:
;   1. Simple: "2,8,1" - creates Linear layers between sizes
;   2. With activations: "2,relu,8,sigmoid,1" - inserts activation after linear
;   3. Full syntax: "Sequential(Linear(2,8), ReLU(), Linear(8,1))"
parse_architecture:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    
    mov r12, rdi                ; architecture string
    mov r14, rsi                ; config pointer
    
    ; Check if it starts with "Sequential("
    mov rdi, r12
    lea rsi, [rel seq_prefix]
    mov ecx, 11                 ; length of "Sequential("
    call str_starts_with
    test eax, eax
    jz .parse_legacy            ; not Sequential, try legacy format
    
    ; Skip "Sequential("
    add r12, 11
    
    ; Check if this is full syntax (contains "Linear(") or simple
    mov rdi, r12
    lea rsi, [rel layer_linear]
    mov ecx, 6                  ; length of "Linear"
    call str_contains
    test eax, eax
    jnz .parse_full_syntax
    
    ; Simple syntax - parse comma-separated numbers with optional activations
    mov rdi, r12
    mov rsi, r14                ; config pointer
    call parse_layer_sizes
    jmp .parse_done
    
.parse_full_syntax:
    ; Full syntax parsing: Sequential(Linear(2,8), ReLU(), Linear(8,1))
    mov rdi, r12
    mov rsi, r14                ; config pointer
    call parse_full_architecture
    jmp .parse_done
    
.parse_legacy:
    ; Parse comma-separated layer sizes (simple format without Sequential prefix)
    mov rdi, r12
    mov rsi, r14                ; config pointer
    call parse_layer_sizes
    jmp .parse_done
    
.parse_error:
    xor eax, eax
    
.parse_done:
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; str_contains - Check if string contains substring
; Arguments:
;   rdi - string to search
;   rsi - substring to find
;   ecx - substring length
; Returns:
;   eax - 1 if contains, 0 otherwise
str_contains:
    push rbx
    push r12
    push r13
    
    mov r12, rdi                ; string
    mov r13, rsi                ; substring
    mov ebx, ecx                ; substring length
    
.search_loop:
    mov al, [r12]
    test al, al
    jz .not_found
    
    ; Try to match at this position
    push r12
    mov rdi, r12
    mov rsi, r13
    mov ecx, ebx
    call str_starts_with
    pop r12
    
    test eax, eax
    jnz .found
    
    inc r12
    jmp .search_loop
    
.found:
    mov eax, 1
    pop r13
    pop r12
    pop rbx
    ret
    
.not_found:
    xor eax, eax
    pop r13
    pop r12
    pop rbx
    ret

; parse_full_architecture - Parse full architecture syntax
; Arguments:
;   rdi - string after "Sequential(" like "Linear(2,8), ReLU(), Linear(8,1))"
;   rsi - config pointer
; Returns:
;   rax - Sequential model pointer
parse_full_architecture:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 128
    
    mov r12, rdi                ; string pointer
    mov r15, rsi                ; config pointer
    mov qword [rbp - 72], 0     ; first in_features (for config)
    mov qword [rbp - 80], 0     ; last out_features (for config)
    mov qword [rbp - 88], 0     ; layer count
    
    ; Create Sequential container
    xor edi, edi
    xor esi, esi
    call neural_sequential_create
    test rax, rax
    jz .full_parse_error
    mov r13, rax                ; sequential pointer
    
.parse_module_loop:
    ; Skip whitespace and commas
    call skip_whitespace_and_comma
    
    ; Check for end
    mov al, [r12]
    test al, al
    jz .full_parse_done
    cmp al, ')'
    je .full_parse_done
    
    ; Extract layer type name
    lea rdi, [rbp - 64]         ; token buffer
    xor ecx, ecx
.extract_layer_name:
    mov al, [r12]
    test al, al
    jz .layer_name_done
    cmp al, '('
    je .layer_name_done
    cmp al, ' '
    je .layer_name_done
    cmp al, ','
    je .layer_name_done
    cmp ecx, 30
    jge .layer_name_done
    mov [rdi + rcx], al
    inc r12
    inc ecx
    jmp .extract_layer_name
    
.layer_name_done:
    mov byte [rdi + rcx], 0     ; null terminate
    
    ; Skip to opening paren
    mov al, [r12]
    cmp al, '('
    jne .skip_to_paren
    jmp .have_paren
    
.skip_to_paren:
    mov al, [r12]
    test al, al
    jz .full_parse_done
    cmp al, '('
    je .have_paren
    inc r12
    jmp .skip_to_paren
    
.have_paren:
    inc r12                     ; skip '('
    
    ; Match layer type and create
    push r12
    push r13
    lea rdi, [rbp - 64]         ; layer name
    call match_layer_type
    pop r13
    pop r12
    
    mov [rbp - 96], eax         ; save layer type
    
    cmp eax, LAYER_LINEAR
    je .create_linear
    cmp eax, LAYER_ACTIVATION
    je .create_activation_layer
    ; Unknown layer type, skip to closing paren
    jmp .skip_to_close_paren
    
.create_linear:
    ; Parse parameters: in_features, out_features
    mov rdi, r12
    call parse_int_from_str
    mov r12, rdi
    mov r14d, eax               ; in_features
    
    ; Skip comma
    call skip_whitespace_and_comma
    
    ; Parse out_features
    mov rdi, r12
    call parse_int_from_str
    mov r12, rdi
    mov ebx, eax                ; out_features
    
    ; Update config if this is the first/last layer
    cmp qword [rbp - 72], 0
    jnz .not_first_linear
    mov [rbp - 72], r14         ; first in_features
.not_first_linear:
    mov [rbp - 80], rbx         ; update last out_features
    inc qword [rbp - 88]        ; layer count
    
    ; Create Linear layer
    push r12
    push r13
    push rbx
    mov edi, r14d               ; in_features
    mov esi, ebx                ; out_features
    xor edx, edx                ; dtype = DT_FLOAT32
    call linear_create
    pop rbx
    pop r13
    pop r12
    
    test rax, rax
    jz .skip_to_close_paren
    
    ; Add to sequential
    push r12
    push r13
    mov rdi, r13                ; sequential
    mov rsi, rax                ; linear layer
    call neural_sequential_add
    pop r13
    pop r12
    jmp .skip_to_close_paren
    
.create_activation_layer:
    ; Get activation type from match
    mov eax, [rbp - 104]        ; activation type stored by match_layer_type
    
    ; Create activation layer
    push r12
    push r13
    mov edi, eax                ; activation type
    xor esi, esi                ; default alpha
    call activation_create
    pop r13
    pop r12
    
    test rax, rax
    jz .skip_to_close_paren
    
    ; Add to sequential
    push r12
    push r13
    mov rdi, r13                ; sequential
    mov rsi, rax                ; activation layer
    call neural_sequential_add
    pop r13
    pop r12
    jmp .skip_to_close_paren
    
.skip_to_close_paren:
    ; Find closing paren
    mov ecx, 1                  ; paren depth
.find_close:
    mov al, [r12]
    test al, al
    jz .full_parse_done
    cmp al, '('
    je .inc_depth
    cmp al, ')'
    je .dec_depth
    inc r12
    jmp .find_close
    
.inc_depth:
    inc ecx
    inc r12
    jmp .find_close
    
.dec_depth:
    dec ecx
    inc r12
    test ecx, ecx
    jnz .find_close
    
    ; Continue to next module
    jmp .parse_module_loop
    
.full_parse_done:
    ; Update config with parsed values
    mov eax, [rbp - 72]         ; input_size
    mov [r15 + OFF_INPUT_SIZE], eax
    mov eax, [rbp - 80]         ; output_size
    mov [r15 + OFF_OUTPUT_SIZE], eax
    mov eax, [rbp - 88]
    mov [r15 + OFF_NUM_LAYERS], eax
    
    mov rax, r13
    add rsp, 128
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    
.full_parse_error:
    xor eax, eax
    add rsp, 128
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; Helper to skip whitespace and commas inline
; Uses r12 as string pointer (modifies it)
skip_whitespace_and_comma:
.skip_loop:
    mov al, [r12]
    test al, al
    jz .skip_done
    cmp al, ' '
    je .skip_next
    cmp al, ','
    je .skip_next
    cmp al, 9                   ; tab
    je .skip_next
    cmp al, 10                  ; newline
    je .skip_next
    jmp .skip_done
.skip_next:
    inc r12
    jmp .skip_loop
.skip_done:
    ret

; match_layer_type - Match layer name to type and activation
; Arguments:
;   rdi - layer name string
; Returns:
;   eax - layer type (LAYER_LINEAR, LAYER_ACTIVATION, etc.)
;   stores activation type at [rbp - 104] if LAYER_ACTIVATION
match_layer_type:
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    
    ; Check for Linear
    mov rdi, r12
    lea rsi, [rel layer_linear]
    call str_equal
    test eax, eax
    jnz .is_linear
    
    ; Check for activation layers
    mov rdi, r12
    lea rsi, [rel layer_relu]
    call str_equal
    test eax, eax
    jnz .is_act_relu
    
    mov rdi, r12
    lea rsi, [rel layer_sigmoid]
    call str_equal
    test eax, eax
    jnz .is_act_sigmoid
    
    mov rdi, r12
    lea rsi, [rel layer_tanh]
    call str_equal
    test eax, eax
    jnz .is_act_tanh
    
    mov rdi, r12
    lea rsi, [rel layer_softmax]
    call str_equal
    test eax, eax
    jnz .is_act_softmax
    
    mov rdi, r12
    lea rsi, [rel layer_gelu]
    call str_equal
    test eax, eax
    jnz .is_act_gelu
    
    mov rdi, r12
    lea rsi, [rel layer_leaky_relu]
    call str_equal
    test eax, eax
    jnz .is_act_leaky_relu
    
    mov rdi, r12
    lea rsi, [rel layer_elu]
    call str_equal
    test eax, eax
    jnz .is_act_elu
    
    mov rdi, r12
    lea rsi, [rel layer_selu]
    call str_equal
    test eax, eax
    jnz .is_act_selu
    
    mov rdi, r12
    lea rsi, [rel layer_swish]
    call str_equal
    test eax, eax
    jnz .is_act_swish
    
    mov rdi, r12
    lea rsi, [rel layer_mish]
    call str_equal
    test eax, eax
    jnz .is_act_mish
    
    mov rdi, r12
    lea rsi, [rel layer_hardswish]
    call str_equal
    test eax, eax
    jnz .is_act_hardswish
    
    mov rdi, r12
    lea rsi, [rel layer_softplus]
    call str_equal
    test eax, eax
    jnz .is_act_softplus
    
    mov rdi, r12
    lea rsi, [rel layer_hardtanh]
    call str_equal
    test eax, eax
    jnz .is_act_hardtanh
    
    ; Also check lowercase activations
    mov rdi, r12
    lea rsi, [rel act_relu]
    call str_equal
    test eax, eax
    jnz .is_act_relu
    
    mov rdi, r12
    lea rsi, [rel act_sigmoid]
    call str_equal
    test eax, eax
    jnz .is_act_sigmoid
    
    mov rdi, r12
    lea rsi, [rel act_tanh]
    call str_equal
    test eax, eax
    jnz .is_act_tanh
    
    ; Unknown layer type
    xor eax, eax
    jmp .match_layer_done
    
.is_linear:
    mov eax, LAYER_LINEAR
    jmp .match_layer_done
    
.is_act_relu:
    mov dword [rbp - 104], ACT_RELU
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_sigmoid:
    mov dword [rbp - 104], ACT_SIGMOID
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_tanh:
    mov dword [rbp - 104], ACT_TANH
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_softmax:
    mov dword [rbp - 104], ACT_SOFTMAX
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_gelu:
    mov dword [rbp - 104], ACT_GELU
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_leaky_relu:
    mov dword [rbp - 104], ACT_LEAKY_RELU
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_elu:
    mov dword [rbp - 104], ACT_ELU
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_selu:
    mov dword [rbp - 104], ACT_SELU
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_swish:
    mov dword [rbp - 104], ACT_SWISH
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_mish:
    mov dword [rbp - 104], ACT_MISH
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_hardswish:
    mov dword [rbp - 104], ACT_HARDSWISH
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_softplus:
    mov dword [rbp - 104], ACT_SOFTPLUS
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.is_act_hardtanh:
    mov dword [rbp - 104], ACT_HARDTANH
    mov eax, LAYER_ACTIVATION
    jmp .match_layer_done
    
.match_layer_done:
    pop r13
    pop r12
    pop rbx
    ret

; parse_layer_sizes - Parse comma-separated layer sizes with optional activations
; Arguments:
;   rdi - string like "10,64,relu,64,sigmoid,10" or "10,64,64,10"
;   rsi - config pointer
; Returns:
;   rax - model pointer
; 
; Enhanced syntax supports:
;   - Numbers only: "2,8,1" creates Linear layers with no activations
;   - With activations: "2,relu,8,sigmoid,1" inserts activation after linear
;   - Supported activations: relu, sigmoid, tanh, softmax, gelu, leaky_relu, 
;                            elu, selu, swish, mish, hardswish, softplus, hardtanh
parse_layer_sizes:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 192                ; increased stack for tokens
    
    ; Layout:
    ; [rbp-80]  = layer sizes array (up to 8 layers, 32 bytes)
    ; [rbp-112] = activation types after each linear (up to 8, 32 bytes)
    ; [rbp-144] = token buffer (32 bytes)
    ; [rbp-176] = saved registers area
    
    mov r12, rdi                ; string pointer
    mov r15, rsi                ; config pointer
    lea r13, [rbp - 80]         ; layer sizes array
    lea rbx, [rbp - 112]        ; activation types array
    xor r14d, r14d              ; layer count (number sizes)
    
    ; Initialize activation types to ACT_NONE
    mov ecx, 8
    lea rdi, [rbp - 112]
.init_act_loop:
    mov dword [rdi], ACT_NONE
    add rdi, 4
    dec ecx
    jnz .init_act_loop
    
    mov qword [rbp - 176], 0    ; current activation index
    
.parse_loop:
    ; Skip whitespace and commas
.skip_space:
    mov al, [r12]
    test al, al
    jz .parse_done
    cmp al, ' '
    je .next_char
    cmp al, ','
    je .next_char
    cmp al, ')'
    je .parse_done
    cmp al, '('
    je .next_char
    
    ; Check if this is a digit or alpha
    cmp al, '0'
    jl .check_alpha
    cmp al, '9'
    jle .parse_number
    
.check_alpha:
    cmp al, 'a'
    jl .next_char
    cmp al, 'z'
    jle .parse_activation
    cmp al, 'A'
    jl .next_char
    cmp al, 'Z'
    jle .parse_activation
    jmp .next_char
    
.next_char:
    inc r12
    jmp .skip_space
    
.parse_number:
    ; Parse integer
    mov rdi, r12
    call parse_int_from_str
    mov [r13 + r14*4], eax      ; store layer size
    mov r12, rdi                ; update string pointer
    inc r14d                    ; increment count
    jmp .parse_loop
    
.parse_activation:
    ; Extract activation name token
    lea rdi, [rbp - 144]        ; token buffer
    xor ecx, ecx                ; token length
.copy_token:
    mov al, [r12]
    test al, al
    jz .token_done
    cmp al, ','
    je .token_done
    cmp al, ')'
    je .token_done
    cmp al, ' '
    je .token_done
    cmp ecx, 30                 ; max token length
    jge .token_done
    mov [rdi + rcx], al
    inc r12
    inc ecx
    jmp .copy_token
    
.token_done:
    mov byte [rdi + rcx], 0     ; null terminate
    
    ; Match activation name
    push r12
    push r13
    push r14
    push rbx
    
    lea rdi, [rbp - 144]        ; token
    call match_activation_name
    
    pop rbx
    pop r14
    pop r13
    pop r12
    
    ; eax now contains activation type (ACT_NONE if not matched)
    test eax, eax
    jz .parse_loop              ; not an activation, skip
    
    ; Store activation type for the previous linear layer
    mov ecx, r14d
    test ecx, ecx
    jz .parse_loop              ; no linear yet
    dec ecx                     ; index of last linear
    mov [rbx + rcx*4], eax      ; store activation type
    jmp .parse_loop
    
.parse_done:
    ; Set config fields from parsed layer sizes
    test r14d, r14d
    jz .parse_error
    
    mov eax, [r13]              ; input_size = first layer
    mov [r15 + OFF_INPUT_SIZE], eax
    mov eax, [r13 + r14*4 - 4]  ; output_size = last layer
    mov [r15 + OFF_OUTPUT_SIZE], eax
    mov eax, r14d
    sub eax, 1                  ; num_layers = transitions
    mov [r15 + OFF_NUM_LAYERS], eax
    ; For simplicity, set hidden_size to the first hidden layer
    cmp r14d, 3
    jb .no_hidden
    mov eax, [r13 + 4]          ; second layer
    mov [r15 + OFF_HIDDEN_SIZE], eax
.no_hidden:
    
    ; Now build the model from layer sizes
    cmp r14d, 2
    jb .parse_error             ; need at least input and output
    
    ; Create Sequential
    push r13
    push r14
    push rbx                    ; activation types array
    
    xor edi, edi                ; NULL modules
    xor esi, esi                ; 0 num_modules
    call neural_sequential_create
    
    pop rbx
    pop r14
    pop r13
    
    test rax, rax
    jz .parse_error
    
    mov [rbp - 176], rax        ; save sequential pointer
    
    ; Add layers with activations
    xor r8d, r8d                ; current index
    mov ecx, r14d
    dec ecx                     ; number of transitions
    
.add_layers:
    cmp r8d, ecx
    jge .add_done
    
    ; Get input and output sizes - save registers before call
    push rcx
    push r8
    push rbx
    push r13
    mov [rbp - 184], rbx        ; save activation array ptr
    
    mov edi, [r13 + r8*4]       ; in_features
    mov esi, [r13 + r8*4 + 4]   ; out_features
    
    ; Create Linear layer
    xor edx, edx                ; dtype = DT_FLOAT32
    call linear_create
    
    pop r13
    pop rbx
    pop r8
    pop rcx
    
    test rax, rax
    jz .parse_error
    
    ; Add Linear to Sequential
    push rcx
    push r8
    push rbx
    push r13
    push rax                    ; save linear layer
    
    mov rdi, [rbp - 176]        ; sequential pointer
    mov rsi, rax                ; linear layer
    call neural_sequential_add
    
    pop rax                     ; restore (not needed but stack balance)
    pop r13
    pop rbx
    pop r8
    pop rcx
    
    ; Check if we need to add activation
    mov eax, [rbx + r8*4]       ; get activation type for this layer
    test eax, eax
    jz .next_layer              ; no activation
    
    ; Create activation layer
    push rcx
    push r8
    push rbx
    push r13
    mov edi, eax                ; activation type
    xor esi, esi                ; default alpha
    call activation_create
    pop r13
    pop rbx
    pop r8
    pop rcx
    
    test rax, rax
    jz .next_layer              ; skip if failed
    
    ; Add activation to Sequential
    push rcx
    push r8
    push rbx
    push r13
    
    mov rdi, [rbp - 176]        ; sequential pointer
    mov rsi, rax                ; activation layer
    call neural_sequential_add
    
    pop r13
    pop rbx
    pop r8
    pop rcx
    
.next_layer:
    inc r8d
    jmp .add_layers
    
.add_done:
    mov rax, [rbp - 176]        ; return sequential pointer
    jmp .cleanup
    
.parse_error:
    xor eax, eax
    
.cleanup:
    add rsp, 192
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; match_activation_name - Match activation name string to type constant
; Arguments:
;   rdi - null-terminated activation name string
; Returns:
;   eax - activation type constant (ACT_NONE if no match)
match_activation_name:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    mov r12, rdi                ; save string pointer
    
    ; Try each activation name
    mov rdi, r12
    lea rsi, [rel act_relu]
    call str_equal
    test eax, eax
    jnz .is_relu
    
    mov rdi, r12
    lea rsi, [rel act_sigmoid]
    call str_equal
    test eax, eax
    jnz .is_sigmoid
    
    mov rdi, r12
    lea rsi, [rel act_tanh]
    call str_equal
    test eax, eax
    jnz .is_tanh
    
    mov rdi, r12
    lea rsi, [rel act_softmax]
    call str_equal
    test eax, eax
    jnz .is_softmax
    
    mov rdi, r12
    lea rsi, [rel act_gelu]
    call str_equal
    test eax, eax
    jnz .is_gelu
    
    mov rdi, r12
    lea rsi, [rel act_leaky_relu]
    call str_equal
    test eax, eax
    jnz .is_leaky_relu
    
    mov rdi, r12
    lea rsi, [rel act_elu]
    call str_equal
    test eax, eax
    jnz .is_elu
    
    mov rdi, r12
    lea rsi, [rel act_selu]
    call str_equal
    test eax, eax
    jnz .is_selu
    
    mov rdi, r12
    lea rsi, [rel act_swish]
    call str_equal
    test eax, eax
    jnz .is_swish
    
    mov rdi, r12
    lea rsi, [rel act_mish]
    call str_equal
    test eax, eax
    jnz .is_mish
    
    mov rdi, r12
    lea rsi, [rel act_hardswish]
    call str_equal
    test eax, eax
    jnz .is_hardswish
    
    mov rdi, r12
    lea rsi, [rel act_softplus]
    call str_equal
    test eax, eax
    jnz .is_softplus
    
    mov rdi, r12
    lea rsi, [rel act_hardtanh]
    call str_equal
    test eax, eax
    jnz .is_hardtanh
    
    ; No match
    xor eax, eax
    jmp .match_done
    
.is_relu:
    mov eax, ACT_RELU
    jmp .match_done
.is_sigmoid:
    mov eax, ACT_SIGMOID
    jmp .match_done
.is_tanh:
    mov eax, ACT_TANH
    jmp .match_done
.is_softmax:
    mov eax, ACT_SOFTMAX
    jmp .match_done
.is_gelu:
    mov eax, ACT_GELU
    jmp .match_done
.is_leaky_relu:
    mov eax, ACT_LEAKY_RELU
    jmp .match_done
.is_elu:
    mov eax, ACT_ELU
    jmp .match_done
.is_selu:
    mov eax, ACT_SELU
    jmp .match_done
.is_swish:
    mov eax, ACT_SWISH
    jmp .match_done
.is_mish:
    mov eax, ACT_MISH
    jmp .match_done
.is_hardswish:
    mov eax, ACT_HARDSWISH
    jmp .match_done
.is_softplus:
    mov eax, ACT_SOFTPLUS
    jmp .match_done
.is_hardtanh:
    mov eax, ACT_HARDTANH
    jmp .match_done
    
.match_done:
    pop r12
    pop rbx
    pop rbp
    ret

; str_equal - Compare two null-terminated strings (case-insensitive)
; Arguments:
;   rdi - string 1
;   rsi - string 2
; Returns:
;   eax - 1 if equal, 0 otherwise
str_equal:
    push rbx
.cmp_loop:
    mov al, [rdi]
    mov bl, [rsi]
    
    ; Convert to lowercase
    cmp al, 'A'
    jl .no_lower1
    cmp al, 'Z'
    jg .no_lower1
    add al, 32
.no_lower1:
    cmp bl, 'A'
    jl .no_lower2
    cmp bl, 'Z'
    jg .no_lower2
    add bl, 32
.no_lower2:
    
    cmp al, bl
    jne .not_equal
    
    test al, al
    jz .equal
    
    inc rdi
    inc rsi
    jmp .cmp_loop
    
.equal:
    mov eax, 1
    pop rbx
    ret
    
.not_equal:
    xor eax, eax
    pop rbx
    ret

; str_starts_with - Check if string starts with prefix
; Arguments:
;   rdi - string
;   rsi - prefix
;   ecx - prefix length
; Returns:
;   eax - 1 if starts with, 0 otherwise
str_starts_with:
    push rbx
    xor ebx, ebx                ; index
    
.check_loop:
    cmp ebx, ecx
    je .starts_with
    
    mov al, [rdi + rbx]
    mov dl, [rsi + rbx]
    cmp al, dl
    jne .not_starts_with
    
    inc ebx
    jmp .check_loop
    
.starts_with:
    mov eax, 1
    jmp .done
    
.not_starts_with:
    xor eax, eax
    
.done:
    pop rbx
    ret

; parse_int_from_str - Parse integer from string
; Arguments:
;   rdi - string pointer (updated to point after number)
; Returns:
;   eax - parsed integer
;   rdi - updated string pointer
parse_int_from_str:
    push rbx
    xor eax, eax                ; result
    xor ebx, ebx                ; sign (0=positive)
    
    ; Skip whitespace
.skip_space:
    mov dl, [rdi]
    cmp dl, ' '
    je .next_space
    cmp dl, '-'
    je .negative
    jmp .parse_digits
    
.next_space:
    inc rdi
    jmp .skip_space
    
.negative:
    mov bl, 1                   ; negative
    inc rdi
    
.parse_digits:
    mov dl, [rdi]
    cmp dl, '0'
    jb .done
    cmp dl, '9'
    ja .done
    
    ; Multiply current result by 10
    imul eax, 10
    
    ; Add digit
    sub dl, '0'
    add eax, edx
    
    inc rdi
    jmp .parse_digits
    
.done:
    test bl, bl
    jz .positive
    neg eax
    
.positive:
    pop rbx
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
    mov r12, rsi                ; model
    
    ; Check if this is a legacy model
    mov rax, [rsi + 8]          ; check second field
    cmp rax, 0x1000             ; if < 0x1000, probably capacity (Sequential)
    jb .sequential_optimizer    ; else, probably layer pointer (legacy)
    
    ; Legacy model format
    mov ecx, [r12]              ; num_layers
    xor r13d, r13d              ; total params count
    lea r14, [r12 + 8]          ; layer pointer array
    
.legacy_count_params_loop:
    test ecx, ecx
    jz .legacy_count_params_done
    
    mov rax, [r14]              ; layer pointer
    cmp rax, 100                ; skip markers (1=ReLU, 2=Softmax)
    jb .legacy_count_next_layer
    
    ; Real layer - add its n_params
    add r13d, [rax]             ; module->n_params at offset 0
    
.legacy_count_next_layer:
    add r14, 8
    dec ecx
    jmp .legacy_count_params_loop
    
.legacy_count_params_done:
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
    
.legacy_collect_loop:
    test ecx, ecx
    jz .collect_done
    
    push rcx                    ; save counter
    
    mov rax, [r14]              ; layer pointer
    cmp rax, 100
    jb .legacy_collect_next
    
    ; Module struct:
    ;   offset 0: n_params
    ;   offset 8: params (Tensor**)
    ;   offset 16: param_nodes (Node**)
    mov r12, rax                ; module
    mov ecx, [r12]              ; layer's n_params
    mov rsi, [r12 + 8]          ; params array
    mov rdi, [r12 + 16]         ; param_nodes array
    
.legacy_collect_layer_params:
    test ecx, ecx
    jz .legacy_collect_next
    
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
    jmp .legacy_collect_layer_params
    
.legacy_collect_next:
    pop rcx
    add r14, 8
    dec ecx
    jmp .legacy_collect_loop
    
    jmp .collect_done
    
.sequential_optimizer:
    ; Allocate a temporary large params array (assume max 100 params)
    mov edi, 100*8
    call mem_alloc
    test rax, rax
    jz .opt_error
    mov [rbp-72], rax           ; params array
    
    ; Get parameters from model
    mov rdi, r12
    mov rsi, rax                ; params array
    mov rdx, 100                ; max params
    call neural_sequential_parameters
    test rax, rax
    jz .opt_error
    
    mov [rbp-84], eax           ; save total count
    
    ; Allocate param_nodes array (n_params * 8 bytes)
    mov eax, [rbp-84]
    shl eax, 3
    mov edi, eax
    call mem_alloc
    test rax, rax
    jz .opt_error
    mov [rbp-80], rax           ; param_nodes array
    
    ; Now collect params and param_nodes from Sequential modules
    mov r15, [rbp-72]           ; params array
    mov rbx, [rbp-80]           ; param_nodes array
    
    ; Iterate through Sequential modules
    mov rdi, r12                ; model (Sequential)
    xor r14d, r14d              ; module index
    
.collect_seq_params:
    mov rsi, r14
    call neural_sequential_get
    test rax, rax
    jz .collect_done
    
    mov r13, rax                ; current module
    
    ; Check if module has parameters (activation modules have 0)
    mov eax, [r13]              ; MODULE_N_PARAMS at offset 0
    test eax, eax
    jz .skip_paramless_module   ; skip modules with no params
    
    ; Get params and param_nodes from module
    mov rcx, [r13 + 8]          ; MODULE_PARAMS
    test rcx, rcx
    jz .skip_paramless_module
    mov rdx, [r13 + 16]         ; MODULE_PARAM_NODES
    test rdx, rdx
    jz .skip_paramless_module
    
    ; Copy weight tensor and node
    mov rax, [rcx]              ; weight tensor
    mov [r15], rax
    mov rax, [rdx]              ; weight node
    mov [rbx], rax
    add r15, 8
    add rbx, 8
    
    ; Copy bias tensor and node
    mov rax, [rcx + 8]          ; bias tensor
    mov [r15], rax
    mov rax, [rdx + 8]          ; bias node
    mov [rbx], rax
    add r15, 8
    add rbx, 8
    
.skip_paramless_module:
    inc r14d
    jmp .collect_seq_params
    
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
    mov edx, [rel r12]              ; input_size (n_features) - 32-bit value
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
    mov edx, [rel r12]              ; input_size (32-bit zero extended to rdx)
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
    
    mov eax, [rel r14]              ; num_samples
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
    jz .cleanup_batch           ; skip if NULL
    
    mov rsi, 0                  ; requires_grad = false for input
    call node_create
    test rax, rax
    jz .cleanup_batch
    mov rbx, rax                ; input node
    mov [rbp - 96], rax         ; save input node for cleanup
    
    ; Forward pass through model
    mov rdi, r12                ; model
    mov rsi, rbx                ; input node
    call model_forward
    
    test rax, rax
    jz .cleanup_batch
    mov rbx, rax                ; predictions node
    
    ; Calculate loss - use BCE for binary (output_size=1), CE for multi-class
    mov rdi, rbx                ; predictions node
    mov rsi, [rbp - 72]         ; batch_y tensor (labels)
    
    ; Check output_size from config (offset 8)
    mov eax, [r15 + 8]          ; output_size
    cmp eax, 1
    je .use_bce_loss
    cmp eax, 2                  ; Also use BCE for 2-class (binary)
    je .use_bce_loss
    
    ; Multi-class: use cross_entropy_loss
    call cross_entropy_loss
    jmp .loss_computed
    
.use_bce_loss:
    ; Binary classification: apply sigmoid first, then use bce_loss
    mov rdi, rbx                ; predictions node (logits)
    call node_sigmoid           ; apply sigmoid
    test rax, rax
    jz .cleanup_batch
    mov rbx, rax                ; now probabilities
    mov rdi, rbx                ; probabilities node
    mov rsi, [rbp - 72]         ; batch_y tensor
    call bce_loss

.loss_computed:
    push rax                    ; save loss node
    
    ; Calculate accuracy
    mov rcx, [rel rbx]              ; predictions node -> value (tensor)
    mov rdi, rcx                ; logits tensor
    mov rsi, [rbp - 72]         ; batch_y tensor
    call calculate_batch_accuracy
    
    add [rel correct_count], eax
    mov eax, [rbp - 48]         ; batch_size
    add [rel total_count], eax
    
    pop rax                     ; restore loss node
    
    test rax, rax
    jz .cleanup_batch
    
    ; Save loss node for backward pass
    mov [rbp - 88], rax         ; save loss node at rbp-88
    
    ; rax = loss node - get scalar value and accumulate
    ; NODE_VALUE is at offset 0, TENSOR_DATA is at offset 0
    mov rcx, [rel rax]              ; NODE_VALUE = loss tensor
    test rcx, rcx
    jz .do_backward
    mov rcx, [rel rcx]              ; TENSOR_DATA = data pointer
    test rcx, rcx
    jz .do_backward
    movss xmm0, [rel rcx]           ; load loss value (float32)
    
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
    jz .cleanup_batch
    call rax

.cleanup_batch:
    ; =========================================================================
    ; MEMORY CLEANUP: Free computational graph and batch tensors
    ; This prevents memory leaks that would accumulate over training epochs
    ; =========================================================================
    
    ; Free the computational graph starting from loss node
    ; This recursively frees all intermediate nodes created during forward/backward
    mov rdi, [rbp - 88]         ; loss node (root of computation graph)
    test rdi, rdi
    jz .cleanup_batch_tensors
    call node_free_graph
    mov qword [rbp - 88], 0     ; clear pointer
    
.cleanup_batch_tensors:
    ; Free batch_x tensor (created by dataset_get_batch)
    mov rdi, [rbp - 64]         ; batch_x tensor
    test rdi, rdi
    jz .cleanup_batch_y
    call tensor_free
    mov qword [rbp - 64], 0     ; clear pointer
    
.cleanup_batch_y:
    ; Free batch_y tensor (created by dataset_get_batch)
    mov rdi, [rbp - 72]         ; batch_y tensor
    test rdi, rdi
    jz .next_batch
    call tensor_free
    mov qword [rbp - 72], 0     ; clear pointer
    
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
    movss [rel total_loss], xmm0
    
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
    
    ; Check if this is a legacy model
    mov rax, [rdi + 8]          ; check second field
    cmp rax, 0x1000             ; if < 0x1000, probably capacity (Sequential)
    jb .sequential_model        ; else, probably layer pointer (legacy)
    
    ; Legacy model format
    push rbx
    push r12
    push r13
    sub rsp, 24
    
    mov r12, rdi                ; model
    mov r13, rsi                ; current activation
    
    mov ebx, [rel r12]              ; num_layers
    lea r12, [r12 + 8]          ; layer pointers
    
.forward_loop:
    test ebx, ebx
    jz .forward_done
    
    mov rax, [rel r12]
    
    ; Check layer type
    cmp rax, 1                  ; ReLU marker
    je .apply_relu
    cmp rax, 2                  ; Softmax marker
    je .apply_softmax
    cmp rax, 3                  ; Sigmoid marker
    je .apply_sigmoid
    
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
    jmp .forward_next
    
.apply_sigmoid:
    mov rdi, r13
    call sigmoid_forward
    
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
    jmp .done
    
.sequential_model:
    ; Sequential model - iterate modules and call their forward_fn
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                ; sequential
    mov r13, rsi                ; current input node
    
    ; Get number of modules - size is at offset 8, not 0!
    mov r14, [r12 + 8]          ; size (SEQUENTIAL_SIZE at offset 8)
    xor r15d, r15d              ; module index
    
.seq_forward_loop:
    cmp r15, r14
    jge .seq_forward_done
    
    ; Get module - modules array is at offset 16
    mov rax, [r12 + 16]         ; SEQUENTIAL_MODULES at offset 16
    mov rbx, [rax + r15*8]      ; modules[r15]
    
    ; Check if module has params (is a Linear layer)
    mov eax, [rbx]              ; MODULE_N_PARAMS at offset 0
    test eax, eax
    jz .call_activation_forward
    
    ; Linear layer - use linear_forward
    mov rdi, rbx                ; module
    mov rsi, r13                ; input node
    call linear_forward
    mov r13, rax                ; output becomes next input
    jmp .seq_forward_next
    
.call_activation_forward:
    ; Activation layer - call forward_fn via function pointer
    ; forward_fn signature: (Module* self, Node* input, Node** output)
    mov rdi, rbx                ; module
    mov rsi, r13                ; input node
    lea rdx, [rsp]              ; output pointer (on stack)
    mov rax, [rbx + 24]         ; MODULE_FORWARD_FN at offset 24
    test rax, rax
    jz .seq_forward_next        ; skip if no forward_fn
    call rax
    ; Check for error
    test eax, eax
    jnz .seq_forward_next       ; skip on error
    mov r13, [rsp]              ; get output node
    
.seq_forward_next:
    inc r15
    jmp .seq_forward_loop
    
.seq_forward_done:
    mov rax, r13                ; return final output
    
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    
.done:
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
    mov ecx, [rel rax]              ; batch_size
    mov r14d, ecx
    
    ; Get num_classes
    mov edx, [rax + 8]          ; num_classes
    mov r15d, edx
    
    xor ebx, ebx                ; correct_count = 0
    xor r8d, r8d                ; i = 0
    
    mov r9, [rel r12]               ; logits data
    mov r10, [rel r13]              ; targets data
    
    ; Check for binary classification (single output)
    cmp r15d, 1
    je .binary_acc_loop
    
.acc_loop:
    cmp r8d, r14d
    jge .acc_done
    
    ; Find argmax for sample i
    ; logits[rel i] starts at r9 + i * num_classes * 4 (assuming float32)
    mov eax, r8d
    imul eax, r15d
    shl eax, 2                  ; * 4 bytes
    lea rdi, [r9 + rax]         ; pointer to logits for this sample
    
    ; Find max index in this row
    xor ecx, ecx                ; max_idx = 0
    movss xmm0, [rel rdi]           ; max_val
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
    ; targets[rel i] is at r10 + i * 4
    movss xmm2, [r10 + r8*4]
    cvttss2si eax, xmm2         ; convert float target to int

.check_match:
    cmp ecx, eax
    jne .next_sample
    
    inc ebx                     ; correct!
    
.next_sample:
    inc r8d
    jmp .acc_loop

; Binary classification accuracy loop (output_size = 1)
; Compare sigmoid output > 0.5 with target (0 or 1)
.binary_acc_loop:
    cmp r8d, r14d
    jge .acc_done
    
    ; Get prediction: logits[rel i] (single value per sample)
    movss xmm0, [r9 + r8*4]     ; logits[rel i]
    
    ; Get target: targets[rel i]
    movss xmm1, [r10 + r8*4]    ; targets[rel i]
    
    ; Convert prediction to class: pred > 0.5 ? 1 : 0
    mov eax, 0x3F000000         ; 0.5 in float
    movd xmm2, eax
    comiss xmm0, xmm2
    ja .pred_class_1
    xor ecx, ecx                ; predicted class = 0
    jmp .binary_compare
.pred_class_1:
    mov ecx, 1                  ; predicted class = 1
    
.binary_compare:
    ; Convert target to int class
    cvttss2si eax, xmm1
    
    ; Compare
    cmp ecx, eax
    jne .binary_next
    
    inc ebx                     ; correct!
    
.binary_next:
    inc r8d
    jmp .binary_acc_loop
    
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
    cmp eax, [rel r13]              ; num_samples
    jge .eval_done
    
    ; Calculate batch size (min of 32 and remaining samples)
    mov ecx, [rel r13]
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
    jz .eval_cleanup               ; Skip if forward failed
    mov r15, rax                ; output node
    
    ; Calculate batch accuracy
    mov rdi, r15                ; output node
    mov rdi, [rel rdi]              ; output tensor (logits)
    mov rsi, [rbp - 56]         ; labels tensor
    call calculate_batch_accuracy
    
    add [rbp - 60], eax         ; correct_count += batch_correct
    mov eax, [rbp - 72]
    add [rbp - 64], eax         ; total_count += batch_size
    
.eval_cleanup:
    ; Cleanup: Free computational graph and batch tensors
    ; Free graph starting from output node (if exists)
    test r15, r15
    jz .eval_cleanup_batch
    mov rdi, r15
    call node_free_graph
    
.eval_cleanup_batch:
    ; Free batch_x tensor
    mov rdi, [rbp - 48]
    test rdi, rdi
    jz .eval_cleanup_y
    call tensor_free
    
.eval_cleanup_y:
    ; Free batch_y tensor
    mov rdi, [rbp - 56]
    test rdi, rdi
    jz .eval_next
    call tensor_free
    
.eval_next:
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
    cmp eax, [rel r13]              ; num_samples
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
    lea rdi, [rel msg_result]
    call print_string
    
    ; Find argmax of output
    mov rdi, r15                ; output node
    mov rdi, [rel rdi]              ; output tensor
    call tensor_argmax
    mov rbx, rax                ; prediction
    
    mov rdi, rax
    call print_int
    
    ; Check accuracy if label exists
    mov rax, [rbp - 56]         ; out_y tensor
    test rax, rax
    jz .print_newline
    
    ; Print " | Target: "
    lea rdi, [rel msg_loss]         ; reuse loss message for separator
    call print_string
    
    mov rax, [rbp - 56]
    movss xmm0, [rel rax]           ; target value (float)
    cvttss2si eax, xmm0         ; target int
    
    mov rdi, rax
    call print_int
    
    cmp eax, ebx
    jne .print_newline
    
    inc dword [rbp - 60]        ; correct++
    
.print_newline:
    lea rdi, [rel msg_newline]
    call print_string
    
    inc dword [rbp - 64]        ; total++
    inc dword [rbp - 68]        ; sample++
    jmp .infer_loop
    
.infer_summary:
    ; Print accuracy
    lea rdi, [rel msg_acc]
    call print_string
    
    cvtsi2ss xmm0, dword [rbp - 60]
    cvtsi2ss xmm1, dword [rbp - 64]
    divss xmm0, xmm1
    mov eax, 100
    cvtsi2ss xmm1, eax
    mulss xmm0, xmm1
    
    mov edi, 2
    call print_float
    
    lea rdi, [rel msg_percent]
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
    
    mov rsi, [rel r12]              ; data pointer
    
    xor ebx, ebx                ; max_idx = 0
    movss xmm0, [rel rsi]           ; max_val = data[0]
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
    lea rdi, [rel msg_test_header]
    call print_string
    
    ; Test 1: Create a small tensor and verify basic operations
    lea rdi, [rel msg_test1]
    call print_string
    
    ; Create a 2x3 tensor
    mov qword [rel rsp], 2          ; shape[0]
    mov qword [rsp+8], 3        ; shape[1]
    mov rdi, 2                  ; ndim
    lea rsi, [rel rsp]              ; shape
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
    
    lea rdi, [rel msg_pass]
    call print_string
    jmp .test2
    
.test1_fail:
    lea rdi, [rel msg_fail]
    call print_string
    
.test2:
    ; Test 2: Create linear layer and verify parameter count
    lea rdi, [rel msg_test2]
    call print_string
    
    mov rdi, 4                  ; in_features
    mov rsi, 2                  ; out_features
    xor edx, edx                ; dtype
    call linear_create
    mov r13, rax
    
    test r13, r13
    jz .test2_fail
    
    ; Check n_params = 2 (weight + bias)
    mov eax, [rel r13]
    cmp eax, 2
    jne .test2_fail
    
    lea rdi, [rel msg_pass]
    call print_string
    jmp .test3
    
.test2_fail:
    lea rdi, [rel msg_fail]
    call print_string
    
.test3:
    ; Test 3: Verify autograd node creation
    lea rdi, [rel msg_test3]
    call print_string
    
    ; Create a node from tensor
    mov rdi, r12                ; tensor from test1
    mov rsi, 1                  ; requires_grad
    call node_create
    mov r14, rax
    
    test r14, r14
    jz .test3_fail
    
    lea rdi, [rel msg_pass]
    call print_string
    jmp .test4
    
.test3_fail:
    lea rdi, [rel msg_fail]
    call print_string
    
.test4:
    ; Test 4: Math kernel (element-wise add with tensors)
    lea rdi, [rel msg_test4]
    call print_string
    
    ; Create tensor a (1D, 8 elements)
    mov qword [rel rsp], 8          ; shape[0] = 8
    mov rdi, 1                  ; ndim = 1
    lea rsi, [rel rsp]
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
    mov qword [rel rsp], 8
    mov rdi, 1
    lea rsi, [rel rsp]
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
    mov qword [rel rsp], 8
    mov rdi, 1
    lea rsi, [rel rsp]
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
    mov rax, [rel rax]              ; c->data
    mov eax, [rel rax]              ; c->data[0]
    cmp eax, 0x40400000         ; 3.0f
    jne .test4_fail
    
    lea rdi, [rel msg_pass]
    call print_string
    jmp .tests_done
    
.test4_fail:
    lea rdi, [rel msg_fail]
    call print_string
    
.tests_done:
    ; Print summary
    lea rdi, [rel msg_test_done]
    call print_string
    
    ; Cleanup - free all allocated objects
    ; Free tensor c (if created)
    mov rdi, [rsp+16]
    test rdi, rdi
    jz .skip_clean_c
    call tensor_free
.skip_clean_c:
    
    ; Free tensor b (rbx)
    test rbx, rbx
    jz .skip_clean_b
    mov rdi, rbx
    call tensor_free
.skip_clean_b:
    
    ; Free tensor a (r15)
    test r15, r15
    jz .skip_clean_a
    mov rdi, r15
    call tensor_free
.skip_clean_a:
    
    ; Free autograd node (r14)
    test r14, r14
    jz .skip_clean_node
    mov rdi, r14
    call node_free
.skip_clean_node:
    
    ; Free linear layer (r13) using module_free
    test r13, r13
    jz .skip_clean_linear
    mov rdi, r13
    call module_free
.skip_clean_linear:
    
    ; Free initial tensor (r12)
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
    mov rax, [rel end_time]
    sub rax, [rel start_time]
    
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
