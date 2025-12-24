; config_parser.asm - Configuration File Parser
; Parses INI-style configuration files for network architecture and training params
; Format: [section] key=value with # comments

section .data
    ; Default configuration values
    default_lr:         dd 0.001        ; 0.001
    default_batch_size: dd 32
    default_epochs:     dd 100
    default_momentum:   dd 0.9
    default_beta1:      dd 0.9
    default_beta2:      dd 0.999
    default_epsilon:    dd 0.00000001   ; 1e-8
    default_dropout:    dd 0.5
    
    ; Section names
    section_model:      db "model", 0
    section_training:   db "training", 0
    section_optimizer:  db "optimizer", 0
    section_data:       db "data", 0
    
    ; Key names - Model
    key_input_size:     db "input_size", 0
    key_hidden_size:    db "hidden_size", 0
    key_output_size:    db "output_size", 0
    key_num_layers:     db "num_layers", 0
    key_activation:     db "activation", 0
    key_dropout_rate:   db "dropout_rate", 0
    
    ; Key names - Training
    key_epochs:         db "epochs", 0
    key_batch_size:     db "batch_size", 0
    key_learning_rate:  db "learning_rate", 0
    key_lr:             db "lr", 0
    key_weight_decay:   db "weight_decay", 0
    key_early_stopping: db "early_stopping", 0
    key_patience:       db "patience", 0
    key_lr_step_size:   db "lr_step_size", 0
    key_lr_gamma:       db "lr_gamma", 0
    
    ; Key names - Optimizer
    key_optimizer:      db "type", 0
    key_momentum:       db "momentum", 0
    key_beta1:          db "beta1", 0
    key_beta2:          db "beta2", 0
    key_epsilon:        db "epsilon", 0
    
    ; Key names - Data
    key_train_file:     db "train_file", 0
    key_test_file:      db "test_file", 0
    key_train_label_file: db "train_label_file", 0
    key_test_label_file:  db "test_label_file", 0
    key_val_split:      db "val_split", 0
    key_shuffle:        db "shuffle", 0
    key_normalize:      db "normalize", 0
    
    ; Activation names
    act_relu:           db "relu", 0
    act_sigmoid:        db "sigmoid", 0
    act_tanh:           db "tanh", 0
    act_softmax:        db "softmax", 0
    
    ; Optimizer names
    opt_sgd:            db "sgd", 0
    opt_adam:           db "adam", 0
    
    ; Boolean strings
    str_true:           db "true", 0
    str_false:          db "false", 0
    str_yes:            db "yes", 0
    str_no:             db "no", 0
    str_1:              db "1", 0
    str_0:              db "0", 0
    
    ; Error messages
    err_config_open:    db "Error: Cannot open config file", 10, 0
    err_config_parse:   db "Error: Config parse error at line ", 0
    err_config_section: db "Error: Unknown section: ", 0
    err_config_key:     db "Error: Unknown key: ", 0
    newline:            db 10, 0
    dbg_section:        db "[DBG] Section: ", 0
    dbg_data_key:       db "[DBG] process_data_key: ", 0
    dbg_newline:        db 10, 0

section .bss
    ; Configuration structure (256 bytes)
    config_buffer:      resb 256
    
    ; Line parsing buffers
    line_buffer:        resb 1024
    section_buffer:     resb 64
    key_buffer:         resb 64
    value_buffer:       resb 256
    
    ; Current parse state
    current_section:    resq 1
    current_line:       resd 1
    
    ; File buffer
    file_buffer:        resb 8192

section .text
    global config_parse
    global config_create_default
    global config_get_int
    global config_get_float
    global config_get_string
    global config_get_bool
    global config_set_int
    global config_set_float
    global config_set_string
    global config_free
    global parse_float
    global parse_int
    global str_to_lower
    global str_equals_nocase
    
    extern mem_alloc
    extern mem_free
    extern print_string

; ============================================================================
; CONFIGURATION STRUCTURE
; ============================================================================
; Offset   Size   Field
; 0        4      input_size
; 4        4      hidden_size
; 8        4      output_size
; 12       4      num_layers
; 16       4      activation (enum: 0=relu, 1=sigmoid, 2=tanh, 3=softmax)
; 20       4      dropout_rate (float)
; 24       4      epochs
; 28       4      batch_size
; 32       4      learning_rate (float)
; 36       4      weight_decay (float)
; 40       4      early_stopping (bool)
; 44       4      patience
; 48       4      optimizer_type (enum: 0=sgd, 1=adam)
; 52       4      momentum (float)
; 56       4      beta1 (float)
; 60       4      beta2 (float)
; 64       4      epsilon (float)
; 68       8      train_file (pointer)
; 76       8      test_file (pointer)
; 84       8      train_label_file (pointer)
; 92       8      test_label_file (pointer)
; 100      4      val_split (float)
; 104      4      shuffle (bool)
; 108      4      normalize (bool)
; 112      4      hidden_sizes array (up to 8 hidden layers)
; 144      4      lr_step_size (int) - StepLR scheduler
; 148      4      lr_gamma (float) - StepLR decay factor
; 152      104    reserved
; ============================================================================

CONFIG_SIZE             equ 256
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

; Activation enum
ACT_RELU                equ 0
ACT_SIGMOID             equ 1
ACT_TANH                equ 2
ACT_SOFTMAX             equ 3

; Optimizer enum
OPT_SGD                 equ 0
OPT_ADAM                equ 1

; config_create_default - Create config with default values
; Returns:
;   rax - pointer to config structure
config_create_default:
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Allocate config structure
    mov edi, CONFIG_SIZE
    call mem_alloc
    
    test rax, rax
    jz .default_done
    
    mov rbx, rax
    
    ; Zero the structure
    mov rdi, rbx
    mov ecx, CONFIG_SIZE / 8
    xor eax, eax
    rep stosq
    
    ; Set defaults
    mov dword [rbx + OFF_INPUT_SIZE], 784       ; MNIST-like
    mov dword [rbx + OFF_HIDDEN_SIZE], 128
    mov dword [rbx + OFF_OUTPUT_SIZE], 10
    mov dword [rbx + OFF_NUM_LAYERS], 2
    mov dword [rbx + OFF_ACTIVATION], ACT_RELU
    
    ; dropout_rate = 0.5
    mov eax, 0x3F000000                          ; 0.5 in float
    mov [rbx + OFF_DROPOUT_RATE], eax
    
    mov dword [rbx + OFF_EPOCHS], 100
    mov dword [rbx + OFF_BATCH_SIZE], 32
    
    ; learning_rate = 0.001
    mov eax, 0x3A83126F                          ; 0.001 in float
    mov [rbx + OFF_LEARNING_RATE], eax
    
    ; weight_decay = 0
    mov dword [rbx + OFF_WEIGHT_DECAY], 0
    
    mov dword [rbx + OFF_EARLY_STOPPING], 0
    mov dword [rbx + OFF_PATIENCE], 10
    mov dword [rbx + OFF_OPTIMIZER_TYPE], OPT_SGD
    
    ; momentum = 0.9
    mov eax, 0x3F666666                          ; 0.9 in float
    mov [rbx + OFF_MOMENTUM], eax
    
    ; beta1 = 0.9
    mov [rbx + OFF_BETA1], eax
    
    ; beta2 = 0.999
    mov eax, 0x3F7FBE77                          ; 0.999 in float
    mov [rbx + OFF_BETA2], eax
    
    ; epsilon = 1e-8
    mov eax, 0x322BCC77                          ; 1e-8 in float
    mov [rbx + OFF_EPSILON], eax
    
    ; val_split = 0.1
    mov eax, 0x3DCCCCCD                          ; 0.1 in float
    mov [rbx + OFF_VAL_SPLIT], eax
    
    mov dword [rbx + OFF_SHUFFLE], 1
    mov dword [rbx + OFF_NORMALIZE], 1
    
    mov rax, rbx
    
.default_done:
    pop rbx
    pop rbp
    ret

; config_parse - Parse configuration file
; Arguments:
;   rdi - filename string
; Returns:
;   rax - pointer to config structure, NULL on error
config_parse:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                ; filename
    
    ; Create default config first
    call config_create_default
    
    test rax, rax
    jz .parse_error
    
    mov r13, rax                ; config pointer
    
    ; Open file
    mov rax, 2                  ; sys_open
    mov rdi, r12
    xor esi, esi                ; O_RDONLY
    xor edx, edx
    syscall
    
    test rax, rax
    js .parse_open_error
    
    mov r14, rax                ; fd
    
    ; Read entire file into buffer
    mov rax, 0                  ; sys_read
    mov rdi, r14
    lea rsi, [file_buffer]
    mov rdx, 8192
    syscall
    
    test rax, rax
    js .parse_close_error
    
    mov r15, rax                ; bytes read
    
    ; Null terminate
    lea rdi, [file_buffer]
    mov byte [rdi + r15], 0
    
    ; Close file
    mov rax, 3
    mov rdi, r14
    syscall
    
    ; Parse the buffer
    lea rsi, [file_buffer]
    mov dword [current_line], 0
    mov qword [current_section], 0
    
.parse_loop:
    ; Check for end of buffer
    mov al, [rsi]
    test al, al
    jz .parse_done
    
    inc dword [current_line]
    
    ; Skip leading whitespace
.skip_leading:
    mov al, [rsi]
    cmp al, ' '
    je .next_leading
    cmp al, 9                   ; tab
    je .next_leading
    jmp .check_line
    
.next_leading:
    inc rsi
    jmp .skip_leading
    
.check_line:
    ; Check for empty line or comment
    mov al, [rsi]
    cmp al, 10                  ; newline
    je .next_line
    cmp al, 13                  ; CR
    je .next_line
    cmp al, '#'
    je .skip_comment
    cmp al, ';'
    je .skip_comment
    cmp al, '['
    je .parse_section
    
    ; Parse key=value
    jmp .parse_key_value
    
.skip_comment:
.find_newline:
    mov al, [rsi]
    test al, al
    jz .parse_done
    cmp al, 10
    je .next_line
    inc rsi
    jmp .find_newline
    
.next_line:
    inc rsi
    jmp .parse_loop
    
.parse_section:
    inc rsi                     ; skip '['
    lea rdi, [section_buffer]
    
.copy_section:
    mov al, [rsi]
    cmp al, ']'
    je .section_done
    cmp al, 10
    je .section_error
    test al, al
    jz .section_error
    mov [rdi], al
    inc rsi
    inc rdi
    jmp .copy_section
    
.section_done:
    mov byte [rdi], 0           ; null terminate
    inc rsi                     ; skip ']'
    
    ; Save buffer position - rsi will be clobbered by section comparisons
    push rsi
    
    ; Identify section
    lea rdi, [section_buffer]
    call str_to_lower
    
    lea rdi, [section_buffer]
    lea rsi, [section_model]
    call str_equals_nocase
    test eax, eax
    jnz .set_section_model
    
    lea rdi, [section_buffer]
    lea rsi, [section_training]
    call str_equals_nocase
    test eax, eax
    jnz .set_section_training
    
    lea rdi, [section_buffer]
    lea rsi, [section_optimizer]
    call str_equals_nocase
    test eax, eax
    jnz .set_section_optimizer
    
    lea rdi, [section_buffer]
    lea rsi, [section_data]
    call str_equals_nocase
    test eax, eax
    jnz .set_section_data
    
    ; Unknown section - skip
    pop rsi                     ; restore buffer position
    jmp .find_newline
    
.set_section_model:
    mov qword [current_section], 1
    pop rsi                     ; restore buffer position
    jmp .find_newline
    
.set_section_training:
    mov qword [current_section], 2
    pop rsi                     ; restore buffer position
    jmp .find_newline
    
.set_section_optimizer:
    mov qword [current_section], 3
    pop rsi                     ; restore buffer position
    jmp .find_newline
    
.set_section_data:
    mov qword [current_section], 4
    pop rsi                     ; restore buffer position
    jmp .find_newline
    
.section_error:
    pop rsi                     ; restore buffer position
    jmp .find_newline
    
.parse_key_value:
    ; Copy key
    lea rdi, [key_buffer]
    
.copy_key:
    mov al, [rsi]
    cmp al, '='
    je .key_done
    cmp al, ' '
    je .skip_key_space
    cmp al, 10
    je .kv_error
    test al, al
    jz .parse_done
    mov [rdi], al
    inc rsi
    inc rdi
    jmp .copy_key
    
.skip_key_space:
    inc rsi
    jmp .copy_key
    
.key_done:
    mov byte [rdi], 0
    inc rsi                     ; skip '='
    
    ; Skip whitespace after =
.skip_val_space:
    mov al, [rsi]
    cmp al, ' '
    je .next_val_space
    cmp al, 9
    je .next_val_space
    jmp .copy_value
    
.next_val_space:
    inc rsi
    jmp .skip_val_space
    
.copy_value:
    lea rdi, [value_buffer]
    
.copy_val_loop:
    mov al, [rsi]
    cmp al, 10
    je .value_done
    cmp al, 13
    je .value_done
    cmp al, '#'
    je .value_done
    test al, al
    jz .value_done
    mov [rdi], al
    inc rsi
    inc rdi
    jmp .copy_val_loop
    
.value_done:
    mov byte [rdi], 0
    
    ; Trim trailing whitespace from value
    lea rdi, [value_buffer]
    call trim_trailing
    
    ; Process based on current section
    mov rax, [current_section]
    cmp rax, 1
    je .process_model
    cmp rax, 2
    je .process_training
    cmp rax, 3
    je .process_optimizer
    cmp rax, 4
    je .process_data
    jmp .find_newline
    
.process_model:
    push rsi
    call process_model_key
    pop rsi
    jmp .find_newline
    
.process_training:
    push rsi
    call process_training_key
    pop rsi
    jmp .find_newline
    
.process_optimizer:
    push rsi
    call process_optimizer_key
    pop rsi
    jmp .find_newline
    
.process_data:
    push rsi
    call process_data_key
    pop rsi
    jmp .find_newline
    
.kv_error:
    jmp .next_line
    
.parse_done:
    mov rax, r13
    jmp .parse_cleanup
    
.parse_open_error:
    lea rdi, [err_config_open]
    call print_error_msg
    mov rax, r13                ; return default config
    jmp .parse_cleanup
    
.parse_close_error:
    mov rax, 3
    mov rdi, r14
    syscall
    
.parse_error:
    xor eax, eax
    
.parse_cleanup:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; process_model_key - Handle [model] section keys
; Uses: r13 = config pointer, key_buffer, value_buffer
process_model_key:
    push rbp
    mov rbp, rsp
    push rbx
    
    lea rdi, [key_buffer]
    
    ; Check input_size
    lea rsi, [key_input_size]
    call str_equals_nocase
    test eax, eax
    jnz .set_input_size
    
    ; Check hidden_size
    lea rdi, [key_buffer]
    lea rsi, [key_hidden_size]
    call str_equals_nocase
    test eax, eax
    jnz .set_hidden_size
    
    ; Check output_size
    lea rdi, [key_buffer]
    lea rsi, [key_output_size]
    call str_equals_nocase
    test eax, eax
    jnz .set_output_size
    
    ; Check num_layers
    lea rdi, [key_buffer]
    lea rsi, [key_num_layers]
    call str_equals_nocase
    test eax, eax
    jnz .set_num_layers
    
    ; Check activation
    lea rdi, [key_buffer]
    lea rsi, [key_activation]
    call str_equals_nocase
    test eax, eax
    jnz .set_activation
    
    ; Check dropout_rate
    lea rdi, [key_buffer]
    lea rsi, [key_dropout_rate]
    call str_equals_nocase
    test eax, eax
    jnz .set_dropout_rate
    
    jmp .model_key_done
    
.set_input_size:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_INPUT_SIZE], eax
    jmp .model_key_done
    
.set_hidden_size:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_HIDDEN_SIZE], eax
    jmp .model_key_done
    
.set_output_size:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_OUTPUT_SIZE], eax
    jmp .model_key_done
    
.set_num_layers:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_NUM_LAYERS], eax
    jmp .model_key_done
    
.set_activation:
    lea rdi, [value_buffer]
    call str_to_lower
    
    lea rdi, [value_buffer]
    lea rsi, [act_relu]
    call str_equals_nocase
    test eax, eax
    jnz .act_relu_set
    
    lea rdi, [value_buffer]
    lea rsi, [act_sigmoid]
    call str_equals_nocase
    test eax, eax
    jnz .act_sigmoid_set
    
    lea rdi, [value_buffer]
    lea rsi, [act_tanh]
    call str_equals_nocase
    test eax, eax
    jnz .act_tanh_set
    
    lea rdi, [value_buffer]
    lea rsi, [act_softmax]
    call str_equals_nocase
    test eax, eax
    jnz .act_softmax_set
    
    jmp .model_key_done
    
.act_relu_set:
    mov dword [r13 + OFF_ACTIVATION], ACT_RELU
    jmp .model_key_done
    
.act_sigmoid_set:
    mov dword [r13 + OFF_ACTIVATION], ACT_SIGMOID
    jmp .model_key_done
    
.act_tanh_set:
    mov dword [r13 + OFF_ACTIVATION], ACT_TANH
    jmp .model_key_done
    
.act_softmax_set:
    mov dword [r13 + OFF_ACTIVATION], ACT_SOFTMAX
    jmp .model_key_done
    
.set_dropout_rate:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_DROPOUT_RATE], xmm0
    jmp .model_key_done
    
.model_key_done:
    pop rbx
    pop rbp
    ret

; process_training_key - Handle [training] section keys
process_training_key:
    push rbp
    mov rbp, rsp
    
    lea rdi, [key_buffer]
    
    ; Check epochs
    lea rsi, [key_epochs]
    call str_equals_nocase
    test eax, eax
    jnz .set_epochs
    
    ; Check batch_size
    lea rdi, [key_buffer]
    lea rsi, [key_batch_size]
    call str_equals_nocase
    test eax, eax
    jnz .set_batch_size
    
    ; Check learning_rate
    lea rdi, [key_buffer]
    lea rsi, [key_learning_rate]
    call str_equals_nocase
    test eax, eax
    jnz .set_learning_rate
    
    ; Check lr (alias)
    lea rdi, [key_buffer]
    lea rsi, [key_lr]
    call str_equals_nocase
    test eax, eax
    jnz .set_learning_rate
    
    ; Check weight_decay
    lea rdi, [key_buffer]
    lea rsi, [key_weight_decay]
    call str_equals_nocase
    test eax, eax
    jnz .set_weight_decay
    
    ; Check early_stopping
    lea rdi, [key_buffer]
    lea rsi, [key_early_stopping]
    call str_equals_nocase
    test eax, eax
    jnz .set_early_stopping
    
    ; Check patience
    lea rdi, [key_buffer]
    lea rsi, [key_patience]
    call str_equals_nocase
    test eax, eax
    jnz .set_patience
    
    ; Check lr_step_size
    lea rdi, [key_buffer]
    lea rsi, [key_lr_step_size]
    call str_equals_nocase
    test eax, eax
    jnz .set_lr_step_size
    
    ; Check lr_gamma
    lea rdi, [key_buffer]
    lea rsi, [key_lr_gamma]
    call str_equals_nocase
    test eax, eax
    jnz .set_lr_gamma
    
    jmp .training_key_done
    
.set_epochs:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_EPOCHS], eax
    jmp .training_key_done
    
.set_batch_size:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_BATCH_SIZE], eax
    jmp .training_key_done
    
.set_learning_rate:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_LEARNING_RATE], xmm0
    jmp .training_key_done
    
.set_weight_decay:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_WEIGHT_DECAY], xmm0
    jmp .training_key_done
    
.set_early_stopping:
    lea rdi, [value_buffer]
    call parse_bool
    mov [r13 + OFF_EARLY_STOPPING], eax
    jmp .training_key_done
    
.set_patience:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_PATIENCE], eax
    jmp .training_key_done

.set_lr_step_size:
    lea rdi, [value_buffer]
    call parse_int
    mov [r13 + OFF_LR_STEP_SIZE], eax
    jmp .training_key_done

.set_lr_gamma:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_LR_GAMMA], xmm0
    jmp .training_key_done
    
.training_key_done:
    pop rbp
    ret

; process_optimizer_key - Handle [optimizer] section keys
process_optimizer_key:
    push rbp
    mov rbp, rsp
    
    lea rdi, [key_buffer]
    
    ; Check type
    lea rsi, [key_optimizer]
    call str_equals_nocase
    test eax, eax
    jnz .set_optimizer_type
    
    ; Check momentum
    lea rdi, [key_buffer]
    lea rsi, [key_momentum]
    call str_equals_nocase
    test eax, eax
    jnz .set_momentum
    
    ; Check beta1
    lea rdi, [key_buffer]
    lea rsi, [key_beta1]
    call str_equals_nocase
    test eax, eax
    jnz .set_beta1
    
    ; Check beta2
    lea rdi, [key_buffer]
    lea rsi, [key_beta2]
    call str_equals_nocase
    test eax, eax
    jnz .set_beta2
    
    ; Check epsilon
    lea rdi, [key_buffer]
    lea rsi, [key_epsilon]
    call str_equals_nocase
    test eax, eax
    jnz .set_epsilon
    
    jmp .optimizer_key_done
    
.set_optimizer_type:
    lea rdi, [value_buffer]
    call str_to_lower
    
    lea rdi, [value_buffer]
    lea rsi, [opt_sgd]
    call str_equals_nocase
    test eax, eax
    jnz .opt_sgd_set
    
    lea rdi, [value_buffer]
    lea rsi, [opt_adam]
    call str_equals_nocase
    test eax, eax
    jnz .opt_adam_set
    
    jmp .optimizer_key_done
    
.opt_sgd_set:
    mov dword [r13 + OFF_OPTIMIZER_TYPE], OPT_SGD
    jmp .optimizer_key_done
    
.opt_adam_set:
    mov dword [r13 + OFF_OPTIMIZER_TYPE], OPT_ADAM
    jmp .optimizer_key_done
    
.set_momentum:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_MOMENTUM], xmm0
    jmp .optimizer_key_done
    
.set_beta1:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_BETA1], xmm0
    jmp .optimizer_key_done
    
.set_beta2:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_BETA2], xmm0
    jmp .optimizer_key_done
    
.set_epsilon:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_EPSILON], xmm0
    jmp .optimizer_key_done
    
.optimizer_key_done:
    pop rbp
    ret

; process_data_key - Handle [data] section keys
process_data_key:
    push rbp
    mov rbp, rsp
    push rbx
    push r13
    
    ; Get r13 from caller (it's a callee-saved register, so should still be config pointer)
    ; But save our copy to be safe
    
    lea rdi, [key_buffer]
    
    ; Check train_file
    lea rsi, [key_train_file]
    call str_equals_nocase
    test eax, eax
    jnz .set_train_file
    
    ; Check test_file
    lea rdi, [key_buffer]
    lea rsi, [key_test_file]
    call str_equals_nocase
    test eax, eax
    jnz .set_test_file
    
    ; Check train_label_file
    lea rdi, [key_buffer]
    lea rsi, [key_train_label_file]
    call str_equals_nocase
    test eax, eax
    jnz .set_train_label_file
    
    ; Check test_label_file
    lea rdi, [key_buffer]
    lea rsi, [key_test_label_file]
    call str_equals_nocase
    test eax, eax
    jnz .set_test_label_file
    
    ; Check val_split
    lea rdi, [key_buffer]
    lea rsi, [key_val_split]
    call str_equals_nocase
    test eax, eax
    jnz .set_val_split
    
    ; Check shuffle
    lea rdi, [key_buffer]
    lea rsi, [key_shuffle]
    call str_equals_nocase
    test eax, eax
    jnz .set_shuffle
    
    ; Check normalize
    lea rdi, [key_buffer]
    lea rsi, [key_normalize]
    call str_equals_nocase
    test eax, eax
    jnz .set_normalize
    
    jmp .data_key_done
    
.set_train_file:
    ; Allocate and copy string
    lea rdi, [value_buffer]
    call str_length
    mov ebx, eax
    inc ebx                     ; null terminator
    mov edi, ebx
    call mem_alloc
    
    test rax, rax
    jz .data_key_done
    
    mov [r13 + OFF_TRAIN_FILE], rax
    mov rdi, rax
    lea rsi, [value_buffer]
    call str_copy
    jmp .data_key_done
    
.set_test_file:
    lea rdi, [value_buffer]
    call str_length
    mov ebx, eax
    inc ebx
    mov edi, ebx
    call mem_alloc
    
    test rax, rax
    jz .data_key_done
    
    mov [r13 + OFF_TEST_FILE], rax
    mov rdi, rax
    lea rsi, [value_buffer]
    call str_copy
    jmp .data_key_done
    
.set_train_label_file:
    lea rdi, [value_buffer]
    call str_length
    mov ebx, eax
    inc ebx
    mov edi, ebx
    call mem_alloc
    
    test rax, rax
    jz .data_key_done
    
    mov [r13 + OFF_TRAIN_LABEL_FILE], rax
    mov rdi, rax
    lea rsi, [value_buffer]
    call str_copy
    jmp .data_key_done
    
.set_test_label_file:
    lea rdi, [value_buffer]
    call str_length
    mov ebx, eax
    inc ebx
    mov edi, ebx
    call mem_alloc
    
    test rax, rax
    jz .data_key_done
    
    mov [r13 + OFF_TEST_LABEL_FILE], rax
    mov rdi, rax
    lea rsi, [value_buffer]
    call str_copy
    jmp .data_key_done
    
.set_val_split:
    lea rdi, [value_buffer]
    call parse_float
    movss [r13 + OFF_VAL_SPLIT], xmm0
    jmp .data_key_done
    
.set_shuffle:
    lea rdi, [value_buffer]
    call parse_bool
    mov [r13 + OFF_SHUFFLE], eax
    jmp .data_key_done
    
.set_normalize:
    lea rdi, [value_buffer]
    call parse_bool
    mov [r13 + OFF_NORMALIZE], eax
    jmp .data_key_done
    
.data_key_done:
    pop r13
    pop rbx
    pop rbp
    ret

; parse_int - Parse integer from string
; Arguments:
;   rdi - string pointer
; Returns:
;   eax - parsed integer
parse_int:
    push rbp
    mov rbp, rsp
    
    xor eax, eax                ; result
    xor ecx, ecx                ; sign flag
    
    ; Check for negative
    mov dl, [rdi]
    cmp dl, '-'
    jne .parse_int_loop
    mov ecx, 1
    inc rdi
    
.parse_int_loop:
    mov dl, [rdi]
    cmp dl, '0'
    jb .parse_int_done
    cmp dl, '9'
    ja .parse_int_done
    
    ; result = result * 10 + digit
    imul eax, 10
    sub dl, '0'
    movzx edx, dl
    add eax, edx
    inc rdi
    jmp .parse_int_loop
    
.parse_int_done:
    ; Apply sign
    test ecx, ecx
    jz .parse_int_ret
    neg eax
    
.parse_int_ret:
    pop rbp
    ret

; parse_float - Parse float from string
; Arguments:
;   rdi - string pointer
; Returns:
;   xmm0 - parsed float
parse_float:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    ; Simple float parsing: integer.fraction format
    xorps xmm0, xmm0            ; result
    xorps xmm1, xmm1            ; fraction
    mov dword [rbp - 4], 0      ; sign
    mov dword [rbp - 8], 0      ; in_fraction
    
    ; Check negative
    mov al, [rdi]
    cmp al, '-'
    jne .float_integer
    mov dword [rbp - 4], 1
    inc rdi
    
.float_integer:
    mov al, [rdi]
    cmp al, '.'
    je .float_start_fraction
    cmp al, '0'
    jb .float_apply_sign
    cmp al, '9'
    ja .float_apply_sign
    
    ; result = result * 10 + digit
    mov dword [rbp - 12], 10
    cvtsi2ss xmm2, dword [rbp - 12]
    mulss xmm0, xmm2
    sub al, '0'
    movzx eax, al
    mov [rbp - 12], eax
    cvtsi2ss xmm2, dword [rbp - 12]
    addss xmm0, xmm2
    inc rdi
    jmp .float_integer
    
.float_start_fraction:
    inc rdi                     ; skip '.'
    mov dword [rbp - 12], 10
    cvtsi2ss xmm3, dword [rbp - 12] ; divisor = 10
    
.float_fraction:
    mov al, [rdi]
    cmp al, '0'
    jb .float_apply_sign
    cmp al, '9'
    ja .float_apply_sign
    
    sub al, '0'
    movzx eax, al
    mov [rbp - 12], eax
    cvtsi2ss xmm2, dword [rbp - 12]
    divss xmm2, xmm3            ; digit / divisor
    addss xmm0, xmm2
    
    ; divisor *= 10
    mov dword [rbp - 12], 10
    cvtsi2ss xmm2, dword [rbp - 12]
    mulss xmm3, xmm2
    
    inc rdi
    jmp .float_fraction
    
.float_apply_sign:
    cmp dword [rbp - 4], 0
    je .float_done
    
    ; Negate
    mov dword [rbp - 12], 0x80000000
    movss xmm1, [rbp - 12]
    xorps xmm0, xmm1
    
.float_done:
    add rsp, 16
    pop rbp
    ret

; parse_bool - Parse boolean from string
; Arguments:
;   rdi - string pointer
; Returns:
;   eax - 1 for true, 0 for false
parse_bool:
    push rbp
    mov rbp, rsp
    push r12
    
    mov r12, rdi
    call str_to_lower
    
    ; Check various true values
    mov rdi, r12
    lea rsi, [str_true]
    call str_equals_nocase
    test eax, eax
    jnz .bool_true
    
    mov rdi, r12
    lea rsi, [str_yes]
    call str_equals_nocase
    test eax, eax
    jnz .bool_true
    
    mov rdi, r12
    lea rsi, [str_1]
    call str_equals_nocase
    test eax, eax
    jnz .bool_true
    
    xor eax, eax
    jmp .bool_done
    
.bool_true:
    mov eax, 1
    
.bool_done:
    pop r12
    pop rbp
    ret

; str_to_lower - Convert string to lowercase in place
; Arguments:
;   rdi - string pointer
str_to_lower:
    push rbp
    mov rbp, rsp
    
.lower_loop:
    mov al, [rdi]
    test al, al
    jz .lower_done
    
    cmp al, 'A'
    jb .lower_next
    cmp al, 'Z'
    ja .lower_next
    
    add al, 32                  ; to lowercase
    mov [rdi], al
    
.lower_next:
    inc rdi
    jmp .lower_loop
    
.lower_done:
    pop rbp
    ret

; str_equals_nocase - Case-insensitive string comparison
; Arguments:
;   rdi - string 1
;   rsi - string 2
; Returns:
;   eax - 1 if equal, 0 if not
str_equals_nocase:
    push rbp
    mov rbp, rsp
    
.cmp_loop:
    mov al, [rdi]
    mov cl, [rsi]
    
    ; Convert both to lowercase
    cmp al, 'A'
    jb .no_lower1
    cmp al, 'Z'
    ja .no_lower1
    add al, 32
.no_lower1:
    
    cmp cl, 'A'
    jb .no_lower2
    cmp cl, 'Z'
    ja .no_lower2
    add cl, 32
.no_lower2:
    
    cmp al, cl
    jne .not_equal
    
    test al, al
    jz .equal
    
    inc rdi
    inc rsi
    jmp .cmp_loop
    
.equal:
    mov eax, 1
    jmp .cmp_done
    
.not_equal:
    xor eax, eax
    
.cmp_done:
    pop rbp
    ret

; str_length - Get string length
; Arguments:
;   rdi - string pointer
; Returns:
;   eax - length
str_length:
    push rbp
    mov rbp, rsp
    
    xor eax, eax
    
.len_loop:
    mov cl, [rdi + rax]
    test cl, cl
    jz .len_done
    inc eax
    jmp .len_loop
    
.len_done:
    pop rbp
    ret

; str_copy - Copy string
; Arguments:
;   rdi - destination
;   rsi - source
str_copy:
    push rbp
    mov rbp, rsp
    
.copy_loop:
    mov al, [rsi]
    mov [rdi], al
    test al, al
    jz .copy_done
    inc rdi
    inc rsi
    jmp .copy_loop
    
.copy_done:
    pop rbp
    ret

; trim_trailing - Remove trailing whitespace
; Arguments:
;   rdi - string pointer
trim_trailing:
    push rbp
    mov rbp, rsp
    push rbx
    
    mov rbx, rdi
    
    ; Find end of string
    xor ecx, ecx
.find_end:
    mov al, [rbx + rcx]
    test al, al
    jz .found_end
    inc ecx
    jmp .find_end
    
.found_end:
    dec ecx
    js .trim_done
    
.trim_loop:
    mov al, [rbx + rcx]
    cmp al, ' '
    je .trim_char
    cmp al, 9                   ; tab
    je .trim_char
    cmp al, 10
    je .trim_char
    cmp al, 13
    je .trim_char
    jmp .trim_done
    
.trim_char:
    mov byte [rbx + rcx], 0
    dec ecx
    jns .trim_loop
    
.trim_done:
    pop rbx
    pop rbp
    ret

; config_get_int - Get integer config value
; Arguments:
;   rdi - config pointer
;   esi - offset
; Returns:
;   eax - value
config_get_int:
    mov eax, [rdi + rsi]
    ret

; config_get_float - Get float config value
; Arguments:
;   rdi - config pointer
;   esi - offset
; Returns:
;   xmm0 - value
config_get_float:
    movss xmm0, [rdi + rsi]
    ret

; config_get_string - Get string config value
; Arguments:
;   rdi - config pointer
;   esi - offset
; Returns:
;   rax - string pointer
config_get_string:
    mov rax, [rdi + rsi]
    ret

; config_get_bool - Get boolean config value
; Arguments:
;   rdi - config pointer
;   esi - offset
; Returns:
;   eax - 1 or 0
config_get_bool:
    mov eax, [rdi + rsi]
    ret

; config_set_int - Set integer config value
; Arguments:
;   rdi - config pointer
;   esi - offset
;   edx - value
config_set_int:
    mov [rdi + rsi], edx
    ret

; config_set_float - Set float config value
; Arguments:
;   rdi - config pointer
;   esi - offset
;   xmm0 - value
config_set_float:
    movss [rdi + rsi], xmm0
    ret

; config_set_string - Set string config value
; Arguments:
;   rdi - config pointer
;   esi - offset
;   rdx - string pointer
config_set_string:
    mov [rdi + rsi], rdx
    ret

; config_free - Free config and associated strings
; Arguments:
;   rdi - config pointer
config_free:
    push rbp
    mov rbp, rsp
    push rbx
    
    mov rbx, rdi
    
    ; Free train_file string
    mov rdi, [rbx + OFF_TRAIN_FILE]
    test rdi, rdi
    jz .free_test
    call mem_free
    
.free_test:
    mov rdi, [rbx + OFF_TEST_FILE]
    test rdi, rdi
    jz .free_config
    call mem_free
    
.free_config:
    mov rdi, rbx
    call mem_free
    
    pop rbx
    pop rbp
    ret

; print_error_msg - Print error message
; Arguments:
;   rdi - message string
print_error_msg:
    push rbp
    mov rbp, rsp
    push rbx
    
    mov rbx, rdi
    
    ; Get length
    mov rdi, rbx
    call str_length
    mov edx, eax
    
    mov rax, 1                  ; sys_write
    mov rdi, 2                  ; stderr
    mov rsi, rbx
    syscall
    
    pop rbx
    pop rbp
    ret
