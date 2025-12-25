; =============================================================================
; error.asm - Enhanced Error Handling System
; =============================================================================
; Structured error codes, detailed messages, stack traces
; =============================================================================

section .data
    align 8
    
    ; Error code constants
    ERR_NONE                equ 0
    ERR_NULL_POINTER        equ 1
    ERR_OUT_OF_MEMORY       equ 2
    ERR_INVALID_ARGUMENT    equ 3
    ERR_SHAPE_MISMATCH      equ 4
    ERR_DTYPE_MISMATCH      equ 5
    ERR_FILE_NOT_FOUND      equ 6
    ERR_FILE_READ           equ 7
    ERR_FILE_WRITE          equ 8
    ERR_PARSE_ERROR         equ 9
    ERR_INVALID_CONFIG      equ 10
    ERR_TENSOR_TOO_LARGE    equ 11
    ERR_INVALID_DTYPE       equ 12
    ERR_DIMENSION_MISMATCH  equ 13
    ERR_NOT_IMPLEMENTED     equ 14
    ERR_INTERNAL            equ 15
    ERR_GRAD_CHECK_FAILED   equ 16
    ERR_NAN_DETECTED        equ 17
    ERR_INF_DETECTED        equ 18
    
    ; Error messages
    err_msg_none:           db "No error", 0
    err_msg_null:           db "Null pointer dereference", 0
    err_msg_oom:            db "Out of memory", 0
    err_msg_invalid_arg:    db "Invalid argument", 0
    err_msg_shape:          db "Shape mismatch in tensor operation", 0
    err_msg_dtype:          db "Data type mismatch", 0
    err_msg_file_not_found: db "File not found", 0
    err_msg_file_read:      db "Error reading file", 0
    err_msg_file_write:     db "Error writing file", 0
    err_msg_parse:          db "Parse error in configuration", 0
    err_msg_config:         db "Invalid configuration value", 0
    err_msg_tensor_large:   db "Tensor size exceeds maximum", 0
    err_msg_invalid_dtype:  db "Invalid data type specified", 0
    err_msg_dim_mismatch:   db "Dimension mismatch in operation", 0
    err_msg_not_impl:       db "Feature not implemented", 0
    err_msg_internal:       db "Internal error", 0
    err_msg_grad_check:     db "Gradient check failed", 0
    err_msg_nan:            db "NaN detected in computation", 0
    err_msg_inf:            db "Infinity detected in computation", 0
    
    ; Error message table
    error_messages:
        dq err_msg_none
        dq err_msg_null
        dq err_msg_oom
        dq err_msg_invalid_arg
        dq err_msg_shape
        dq err_msg_dtype
        dq err_msg_file_not_found
        dq err_msg_file_read
        dq err_msg_file_write
        dq err_msg_parse
        dq err_msg_config
        dq err_msg_tensor_large
        dq err_msg_invalid_dtype
        dq err_msg_dim_mismatch
        dq err_msg_not_impl
        dq err_msg_internal
        dq err_msg_grad_check
        dq err_msg_nan
        dq err_msg_inf
    
    NUM_ERROR_CODES         equ 19
    
    ; Formatting
    err_prefix:             db "[ERROR] ", 0
    err_at:                 db " at ", 0
    err_line:               db " line ", 0
    err_details:            db ": ", 0
    err_expected:           db "Expected: ", 0
    err_got:                db ", Got: ", 0
    err_newline:            db 10, 0
    
    ; Tensor error details
    err_tensor_shape:       db "Tensor shape: [", 0
    err_tensor_dtype:       db "], dtype: ", 0
    err_tensor_null:        db "Tensor is NULL", 0
    err_tensor_data_null:   db "Tensor data pointer is NULL", 0
    
    ; File error details
    err_file_path:          db "File: ", 0
    err_errno:              db ", errno: ", 0
    
    ; Dtype names
    dtype_float32:          db "float32", 0
    dtype_float64:          db "float64", 0
    dtype_unknown:          db "unknown", 0

section .bss
    align 8
    ; Last error state
    last_error_code:        resd 1
    last_error_file:        resq 1
    last_error_line:        resd 1
    last_error_detail:      resq 1
    error_context:          resb 256     ; Buffer for error context string

section .text

extern printf
extern fprintf
extern stderr
extern exit
extern sprintf
extern strlen

; Export error functions
global error_set
global error_get
global error_clear
global error_print
global error_panic
global error_check_null
global error_check_tensor
global error_check_shapes
global error_check_dtypes
global error_check_nan
global error_get_message
global error_format_tensor
global check_null_ptr
global check_tensor_valid
global check_shapes_match
global check_dtypes_match

; =============================================================================
; error_set - Set the last error
; Arguments:
;   EDI = error code
;   RSI = filename (optional, can be NULL)
;   EDX = line number
;   RCX = detail string (optional, can be NULL)
; =============================================================================
error_set:
    push rbp
    mov rbp, rsp
    
    mov [last_error_code], edi
    mov [last_error_file], rsi
    mov [last_error_line], edx
    mov [last_error_detail], rcx
    
    pop rbp
    ret

; =============================================================================
; error_get - Get the last error code
; Returns:
;   EAX = error code
; =============================================================================
error_get:
    mov eax, [last_error_code]
    ret

; =============================================================================
; error_clear - Clear the last error
; =============================================================================
error_clear:
    mov dword [last_error_code], ERR_NONE
    mov qword [last_error_file], 0
    mov dword [last_error_line], 0
    mov qword [last_error_detail], 0
    ret

; =============================================================================
; error_get_message - Get error message string for code
; Arguments:
;   EDI = error code
; Returns:
;   RAX = pointer to message string
; =============================================================================
error_get_message:
    push rbp
    mov rbp, rsp
    
    ; Bounds check
    cmp edi, NUM_ERROR_CODES
    jae .unknown
    
    ; Look up in table
    lea rax, [rel error_messages]
    mov rax, [rax + rdi*8]
    jmp .done
    
.unknown:
    lea rax, [rel err_msg_internal]
    
.done:
    pop rbp
    ret

; =============================================================================
; error_print - Print the last error to stderr
; =============================================================================
error_print:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    ; Get stderr
    mov rax, [rel stderr wrt ..got]
    mov r12, [rax]
    
    ; Print prefix
    mov rdi, r12
    lea rsi, [rel err_prefix]
    xor eax, eax
    call fprintf wrt ..plt
    
    ; Get and print error message
    mov edi, [last_error_code]
    call error_get_message
    mov rbx, rax
    
    mov rdi, r12
    mov rsi, rbx
    xor eax, eax
    call fprintf wrt ..plt
    
    ; Print location if available
    mov rax, [last_error_file]
    test rax, rax
    jz .no_location
    
    mov rdi, r12
    lea rsi, [rel err_at]
    xor eax, eax
    call fprintf wrt ..plt
    
    mov rdi, r12
    mov rsi, [last_error_file]
    xor eax, eax
    call fprintf wrt ..plt
    
    mov rdi, r12
    lea rsi, [rel err_line]
    xor eax, eax
    call fprintf wrt ..plt
    
    mov rdi, r12
    lea rsi, [rel fmt_int]
    mov edx, [last_error_line]
    xor eax, eax
    call fprintf wrt ..plt
    
.no_location:
    ; Print detail if available
    mov rax, [last_error_detail]
    test rax, rax
    jz .no_detail
    
    mov rdi, r12
    lea rsi, [rel err_details]
    xor eax, eax
    call fprintf wrt ..plt
    
    mov rdi, r12
    mov rsi, [last_error_detail]
    xor eax, eax
    call fprintf wrt ..plt
    
.no_detail:
    ; Newline
    mov rdi, r12
    lea rsi, [rel err_newline]
    xor eax, eax
    call fprintf wrt ..plt
    
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; error_panic - Print error and exit
; Arguments:
;   EDI = error code
;   RSI = detail string (optional)
; =============================================================================
error_panic:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    ; Set error
    mov [rsp], edi
    mov [rsp+8], rsi
    
    mov edi, [rsp]
    xor esi, esi                ; no file
    xor edx, edx                ; no line
    mov rcx, [rsp+8]
    call error_set
    
    ; Print error
    call error_print
    
    ; Exit with error code
    mov edi, 1
    call exit wrt ..plt

; =============================================================================
; check_null_ptr - Check if pointer is NULL, set error if so
; Arguments:
;   RDI = pointer to check
;   RSI = context string (what the pointer is)
; Returns:
;   EAX = 1 if valid (not null), 0 if null
; =============================================================================
check_null_ptr:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .is_null
    
    mov eax, 1
    jmp .done
    
.is_null:
    push rsi
    mov edi, ERR_NULL_POINTER
    xor esi, esi
    xor edx, edx
    pop rcx                     ; detail = context string
    call error_set
    
    xor eax, eax
    
.done:
    pop rbp
    ret

; =============================================================================
; check_tensor_valid - Validate a tensor pointer and its data
; Arguments:
;   RDI = Tensor* to check
; Returns:
;   EAX = 1 if valid, 0 if invalid (error set)
; =============================================================================
check_tensor_valid:
    push rbp
    mov rbp, rsp
    push rbx
    
    mov rbx, rdi
    
    ; Check tensor pointer
    test rbx, rbx
    jz .tensor_null
    
    ; Check data pointer
    mov rax, [rbx]              ; TENSOR_DATA
    test rax, rax
    jz .data_null
    
    ; Check valid ndim (1-4)
    mov rax, [rbx + 8]          ; TENSOR_NDIM
    test rax, rax
    jz .invalid_ndim
    cmp rax, 4
    ja .invalid_ndim
    
    ; Check valid dtype
    mov eax, [rbx + 32]         ; TENSOR_DTYPE
    cmp eax, 2                  ; max dtype
    ja .invalid_dtype
    
    mov eax, 1
    jmp .done
    
.tensor_null:
    mov edi, ERR_NULL_POINTER
    xor esi, esi
    xor edx, edx
    lea rcx, [rel err_tensor_null]
    call error_set
    xor eax, eax
    jmp .done
    
.data_null:
    mov edi, ERR_NULL_POINTER
    xor esi, esi
    xor edx, edx
    lea rcx, [rel err_tensor_data_null]
    call error_set
    xor eax, eax
    jmp .done
    
.invalid_ndim:
    mov edi, ERR_INVALID_ARGUMENT
    xor esi, esi
    xor edx, edx
    xor ecx, ecx
    call error_set
    xor eax, eax
    jmp .done
    
.invalid_dtype:
    mov edi, ERR_INVALID_DTYPE
    xor esi, esi
    xor edx, edx
    xor ecx, ecx
    call error_set
    xor eax, eax
    
.done:
    pop rbx
    pop rbp
    ret

; =============================================================================
; check_shapes_match - Check if two tensors have matching shapes
; Arguments:
;   RDI = Tensor* a
;   RSI = Tensor* b
; Returns:
;   EAX = 1 if shapes match, 0 if not (error set)
; =============================================================================
check_shapes_match:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    mov r13, rsi
    
    ; Validate both tensors first
    mov rdi, r12
    call check_tensor_valid
    test eax, eax
    jz .done
    
    mov rdi, r13
    call check_tensor_valid
    test eax, eax
    jz .done
    
    ; Check ndim matches
    mov rax, [r12 + 8]
    cmp rax, [r13 + 8]
    jne .shape_mismatch
    
    ; Check each dimension
    mov rcx, rax                ; ndim
    mov rsi, [r12 + 16]         ; a->shape
    mov rdi, [r13 + 16]         ; b->shape
    
.check_dims:
    test rcx, rcx
    jz .match
    
    mov rax, [rsi]
    cmp rax, [rdi]
    jne .shape_mismatch
    
    add rsi, 8
    add rdi, 8
    dec rcx
    jmp .check_dims
    
.match:
    mov eax, 1
    jmp .done
    
.shape_mismatch:
    mov edi, ERR_SHAPE_MISMATCH
    xor esi, esi
    xor edx, edx
    xor ecx, ecx
    call error_set
    xor eax, eax
    
.done:
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; check_dtypes_match - Check if two tensors have matching dtypes
; Arguments:
;   RDI = Tensor* a
;   RSI = Tensor* b
; Returns:
;   EAX = 1 if dtypes match, 0 if not (error set)
; =============================================================================
check_dtypes_match:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    
    mov r12, rdi
    mov r13, rsi
    
    ; Validate tensors
    mov rdi, r12
    call check_tensor_valid
    test eax, eax
    jz .done
    
    mov rdi, r13
    call check_tensor_valid
    test eax, eax
    jz .done
    
    ; Check dtype
    mov eax, [r12 + 32]
    cmp eax, [r13 + 32]
    jne .dtype_mismatch
    
    mov eax, 1
    jmp .done
    
.dtype_mismatch:
    mov edi, ERR_DTYPE_MISMATCH
    xor esi, esi
    xor edx, edx
    xor ecx, ecx
    call error_set
    xor eax, eax
    
.done:
    pop r13
    pop r12
    pop rbp
    ret

; =============================================================================
; error_check_nan - Check tensor for NaN values
; Arguments:
;   RDI = Tensor*
; Returns:
;   EAX = 1 if no NaN, 0 if NaN found (error set)
; =============================================================================
error_check_nan:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    mov r12, rdi
    
    ; Validate tensor
    call check_tensor_valid
    test eax, eax
    jz .done
    
    ; Get element count and data
    mov rdi, r12
    call tensor_numel wrt ..plt
    mov r13, rax
    
    mov rbx, [r12]              ; data pointer
    mov eax, [r12 + 32]         ; dtype
    
    test eax, eax
    jnz .check_f64
    
    ; Float32 check
.check_f32:
    test r13, r13
    jz .no_nan
    
    movss xmm0, [rbx]
    ucomiss xmm0, xmm0          ; NaN != NaN
    jp .nan_found
    
    add rbx, 4
    dec r13
    jmp .check_f32
    
.check_f64:
    test r13, r13
    jz .no_nan
    
    movsd xmm0, [rbx]
    ucomisd xmm0, xmm0
    jp .nan_found
    
    add rbx, 8
    dec r13
    jmp .check_f64
    
.nan_found:
    mov edi, ERR_NAN_DETECTED
    xor esi, esi
    xor edx, edx
    xor ecx, ecx
    call error_set
    xor eax, eax
    jmp .done
    
.no_nan:
    mov eax, 1
    
.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

section .data
    fmt_int:    db "%d", 0

section .text
extern tensor_numel
; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
