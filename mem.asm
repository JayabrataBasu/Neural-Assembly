; =============================================================================
; mem.asm - Memory Management for Deep Learning Framework
; =============================================================================
; Uses libc malloc/free for memory allocation
; Provides aligned allocation for SIMD operations
; =============================================================================

section .data
    align 8
    err_alloc_failed: db "Error: Memory allocation failed", 10, 0
    err_null_ptr:     db "Error: Null pointer in mem_free", 10, 0

section .bss
    align 8

section .text

; External libc functions
extern malloc
extern free
extern posix_memalign

; Export our functions
global mem_alloc
global mem_free
global mem_alloc_aligned
global mem_zero
global mem_copy

; =============================================================================
; mem_alloc - Allocate memory
; Arguments:
;   RDI = size (uint64_t)
; Returns:
;   RAX = pointer to allocated memory, or NULL on failure
; =============================================================================
mem_alloc:
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Save size
    mov rbx, rdi
    
    ; Call malloc(size)
    ; RDI already contains size
    sub rsp, 8              ; Align stack to 16 bytes
    call malloc wrt ..plt
    add rsp, 8
    
    ; Check for NULL
    test rax, rax
    jz .alloc_failed
    
    pop rbx
    pop rbp
    ret

.alloc_failed:
    ; Return NULL (rax is already 0)
    pop rbx
    pop rbp
    ret

; =============================================================================
; mem_free - Free allocated memory
; Arguments:
;   RDI = pointer to memory to free
; Returns:
;   nothing
; =============================================================================
mem_free:
    push rbp
    mov rbp, rsp
    
    ; Check for NULL pointer
    test rdi, rdi
    jz .done
    
    ; Call free(ptr)
    sub rsp, 8              ; Align stack
    call free wrt ..plt
    add rsp, 8
    
.done:
    pop rbp
    ret

; =============================================================================
; mem_alloc_aligned - Allocate aligned memory
; Arguments:
;   RDI = size (uint64_t)
;   RSI = alignment (uint64_t) - must be power of 2 and >= sizeof(void*)
; Returns:
;   RAX = pointer to aligned memory, or NULL on failure
; =============================================================================
mem_alloc_aligned:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 24             ; Space for local variable + alignment
    
    ; Save parameters
    mov rbx, rdi            ; size
    mov r12, rsi            ; alignment
    
    ; posix_memalign(void **memptr, size_t alignment, size_t size)
    lea rdi, [rsp]          ; memptr - address of local variable
    mov rsi, r12            ; alignment
    mov rdx, rbx            ; size
    call posix_memalign wrt ..plt
    
    ; Check return value (0 = success)
    test eax, eax
    jnz .aligned_alloc_failed
    
    ; Get the allocated pointer
    mov rax, [rsp]
    
    add rsp, 24
    pop r12
    pop rbx
    pop rbp
    ret

.aligned_alloc_failed:
    xor eax, eax            ; Return NULL
    add rsp, 24
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; mem_zero - Zero out memory region
; Arguments:
;   RDI = pointer to memory
;   RSI = size in bytes
; Returns:
;   nothing
; =============================================================================
mem_zero:
    push rbp
    mov rbp, rsp
    push rdi
    push rcx
    
    mov rcx, rsi            ; count
    xor eax, eax            ; value = 0
    rep stosb               ; Fill with zeros
    
    pop rcx
    pop rdi
    pop rbp
    ret

; =============================================================================
; mem_copy - Copy memory
; Arguments:
;   RDI = destination pointer
;   RSI = source pointer
;   RDX = size in bytes
; Returns:
;   RAX = destination pointer
; =============================================================================
mem_copy:
    push rbp
    mov rbp, rsp
    push rdi
    push rsi
    push rcx
    
    mov rax, rdi            ; Save destination for return
    mov rcx, rdx            ; count
    rep movsb               ; Copy bytes
    
    pop rcx
    pop rsi
    pop rdi
    pop rbp
    ret
