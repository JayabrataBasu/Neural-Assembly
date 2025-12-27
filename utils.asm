; =============================================================================
; utils.asm - Utility Functions for Deep Learning Framework
; =============================================================================
; Logging, error handling, assertions, string utilities
; =============================================================================

section .data
    align 8
    newline:        db 10, 0
    assert_prefix:  db "Assertion failed: ", 0
    info_prefix:    db "[rel INFO] ", 0
    
    ; Format strings for printing
    fmt_int:        db "%ld", 0
    fmt_float:      db "%f", 0
    fmt_string:     db "%s", 0
    fmt_newline:    db 10, 0

section .bss
    align 8
    float_buffer:   resb 64     ; Buffer for float to string conversion

section .text

; External libc functions
extern write
extern exit
extern strlen
extern strcmp
extern strcpy
extern strtol
extern strtod
extern printf
extern fprintf
extern stderr

; Import from mem.asm
extern mem_alloc
extern mem_free

; Export our functions
global panic
global log_info
global log_int
global log_float
global assert_true
global str_cmp
global str_len
global str_copy
global str_to_int
global str_to_float
global print_string
global print_int
global print_float

; =============================================================================
; panic - Print error message and exit with non-zero status
; Arguments:
;   RDI = pointer to error message (null-terminated)
; Returns:
;   Does not return
; =============================================================================
panic:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8              ; Align stack
    
    mov rbx, rdi            ; Save message pointer
    
    ; Get stderr file handle
    mov rdi, [rel stderr wrt ..got]
    mov rdi, [rel rdi]
    
    ; Print message to stderr
    mov rsi, rbx
    xor eax, eax
    call fprintf wrt ..plt
    
    ; Print newline
    mov rdi, [rel stderr wrt ..got]
    mov rdi, [rel rdi]
    lea rsi, [rel fmt_newline]
    xor eax, eax
    call fprintf wrt ..plt
    
    ; Exit with status 1
    mov edi, 1
    call exit wrt ..plt
    
    ; Should never reach here
    add rsp, 8
    pop rbx
    pop rbp
    ret

; =============================================================================
; log_info - Print info message to stdout
; Arguments:
;   RDI = pointer to message (null-terminated)
; Returns:
;   nothing
; =============================================================================
log_info:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 8
    
    mov rbx, rdi            ; Save message
    
    ; Print prefix
    lea rdi, [rel info_prefix]
    xor eax, eax
    call printf wrt ..plt
    
    ; Print message
    mov rdi, rbx
    xor eax, eax
    call printf wrt ..plt
    
    ; Print newline
    lea rdi, [rel fmt_newline]
    xor eax, eax
    call printf wrt ..plt
    
    add rsp, 8
    pop rbx
    pop rbp
    ret

; =============================================================================
; log_int - Print an integer to stdout
; Arguments:
;   RDI = integer value
; Returns:
;   nothing
; =============================================================================
log_int:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    mov rsi, rdi            ; value
    lea rdi, [rel fmt_int]
    xor eax, eax
    call printf wrt ..plt
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; log_float - Print a float to stdout
; Arguments:
;   XMM0 = float value (double)
; Returns:
;   nothing
; =============================================================================
log_float:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    lea rdi, [rel fmt_float]
    mov eax, 1              ; 1 float argument in xmm0
    call printf wrt ..plt
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; assert_true - Assert condition is true, panic if not
; Arguments:
;   RDI = condition (0 = false, non-zero = true)
;   RSI = pointer to error message
; Returns:
;   nothing (or does not return if assertion fails)
; =============================================================================
assert_true:
    push rbp
    mov rbp, rsp
    
    test rdi, rdi
    jz .assertion_failed
    
    pop rbp
    ret

.assertion_failed:
    mov rdi, rsi            ; Error message
    call panic              ; Does not return

; =============================================================================
; str_cmp - Compare two strings
; Arguments:
;   RDI = pointer to first string
;   RSI = pointer to second string
; Returns:
;   RAX = 0 if equal, <0 if s1 < s2, >0 if s1 > s2
; =============================================================================
str_cmp:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    call strcmp wrt ..plt
    ; Result already in EAX
    movsx rax, eax
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; str_len - Get string length
; Arguments:
;   RDI = pointer to string
; Returns:
;   RAX = length of string (not including null terminator)
; =============================================================================
str_len:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    call strlen wrt ..plt
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; str_copy - Copy string
; Arguments:
;   RDI = destination pointer
;   RSI = source pointer
; Returns:
;   RAX = destination pointer
; =============================================================================
str_copy:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    call strcpy wrt ..plt
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; str_to_int - Convert string to integer
; Arguments:
;   RDI = pointer to string
; Returns:
;   RAX = integer value
; =============================================================================
str_to_int:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    ; strtol(str, NULL, 10)
    xor esi, esi            ; endptr = NULL
    mov edx, 10             ; base = 10
    call strtol wrt ..plt
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; str_to_float - Convert string to double
; Arguments:
;   RDI = pointer to string
; Returns:
;   XMM0 = double value
; =============================================================================
str_to_float:
    push rbp
    mov rbp, rsp
    sub rsp, 8              ; Align stack to 16 bytes
    
    ; strtod(str, NULL)
    xor esi, esi            ; endptr = NULL
    call strtod wrt ..plt
    ; Result in XMM0
    
    add rsp, 8
    pop rbp
    ret

; =============================================================================
; print_string - Print string to stdout
; Arguments:
;   RDI = pointer to string
; Returns:
;   nothing
; =============================================================================
print_string:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    ; printf("%s", str)
    mov rsi, rdi
    lea rdi, [rel fmt_string]
    xor eax, eax
    call printf wrt ..plt
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; print_int - Print integer to stdout
; Arguments:
;   RDI = integer value
; Returns:
;   nothing
; =============================================================================
print_int:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    mov rsi, rdi
    lea rdi, [rel fmt_int]
    xor eax, eax
    call printf wrt ..plt
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; print_float - Print float to stdout
; Arguments:
;   XMM0 = double value
; Returns:
;   nothing
; =============================================================================
print_float:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    lea rdi, [rel fmt_float]
    mov eax, 1
    call printf wrt ..plt
    
    add rsp, 16
    pop rbp
    ret
; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
