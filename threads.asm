; =============================================================================
; threads.asm - Multi-threading Support with pthreads
; =============================================================================
; Thread pool, parallel batch processing, work stealing
; =============================================================================

section .data
    align 8
    
    ; Thread pool configuration
    MAX_THREADS     equ 32
    WORK_QUEUE_SIZE equ 1024
    
    ; Thread pool state
    pool_initialized:   dd 0
    num_workers:        dd 0
    shutdown_flag:      dd 0
    
    ; Work queue state
    queue_head:         dq 0
    queue_tail:         dq 0
    queue_count:        dq 0
    
    ; Synchronization
    queue_mutex:        dq 0
    queue_cond:         dq 0
    done_cond:          dq 0
    active_tasks:       dq 0
    
    ; Thread function pointers
    parallel_func_ptr:  dq 0
    
    ; Messages
    msg_pool_init:      db "[THREADS] Thread pool initialized with %d workers", 10, 0
    msg_pool_shutdown:  db "[THREADS] Thread pool shutdown", 10, 0
    msg_task_submit:    db "[THREADS] Task submitted", 10, 0

section .bss
    align 8
    
    ; Thread handles
    worker_threads:     resq MAX_THREADS
    
    ; Work queue (circular buffer)
    work_queue:         resq WORK_QUEUE_SIZE * 4  ; Each work item: func, arg1, arg2, arg3
    
    ; Mutex and condition variables (pthread structures)
    ; Using space for pthread_mutex_t (40 bytes) and pthread_cond_t (48 bytes)
    mutex_storage:      resb 64
    cond_storage:       resb 64
    done_cond_storage:  resb 64

section .text

; External pthread functions
extern pthread_create
extern pthread_join
extern pthread_mutex_init
extern pthread_mutex_destroy
extern pthread_mutex_lock
extern pthread_mutex_unlock
extern pthread_cond_init
extern pthread_cond_destroy
extern pthread_cond_wait
extern pthread_cond_signal
extern pthread_cond_broadcast
extern printf
extern sysconf

; Export functions
global thread_pool_init
global thread_pool_shutdown
global thread_pool_submit
global thread_pool_wait
global parallel_for
global parallel_map
global get_num_cpus

; =============================================================================
; get_num_cpus - Get number of CPU cores
; Returns:
;   EAX = number of CPU cores
; =============================================================================
get_num_cpus:
    push rbp
    mov rbp, rsp
    
    ; sysconf(_SC_NPROCESSORS_ONLN) = 84
    mov edi, 84
    call sysconf
    
    ; Ensure at least 1
    cmp eax, 1
    jge .done
    mov eax, 1
    
.done:
    pop rbp
    ret

; =============================================================================
; thread_pool_init - Initialize thread pool
; Arguments:
;   EDI = number of threads (0 = auto-detect)
; Returns:
;   EAX = 0 on success, -1 on failure
; =============================================================================
thread_pool_init:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 24
    
    ; Check if already initialized
    cmp dword [pool_initialized], 1
    je .already_init
    
    ; Determine number of workers
    mov r12d, edi
    test r12d, r12d
    jnz .has_count
    
    ; Auto-detect
    call get_num_cpus
    mov r12d, eax
    
.has_count:
    ; Clamp to MAX_THREADS
    cmp r12d, MAX_THREADS
    jle .count_ok
    mov r12d, MAX_THREADS
    
.count_ok:
    mov [num_workers], r12d
    
    ; Initialize mutex
    lea rdi, [mutex_storage]
    xor esi, esi                ; NULL for default attributes
    call pthread_mutex_init
    test eax, eax
    jnz .mutex_fail
    
    ; Initialize condition variable
    lea rdi, [cond_storage]
    xor esi, esi
    call pthread_cond_init
    test eax, eax
    jnz .cond_fail
    
    ; Initialize done condition
    lea rdi, [done_cond_storage]
    xor esi, esi
    call pthread_cond_init
    test eax, eax
    jnz .done_cond_fail
    
    ; Reset queue
    mov qword [queue_head], 0
    mov qword [queue_tail], 0
    mov qword [queue_count], 0
    mov qword [active_tasks], 0
    mov dword [shutdown_flag], 0
    
    ; Create worker threads
    xor r13d, r13d              ; Thread index
    
.create_loop:
    cmp r13d, r12d
    jge .threads_created
    
    lea rdi, [worker_threads + r13*8]  ; &threads[i]
    xor esi, esi                        ; NULL attributes
    lea rdx, [worker_routine]           ; Start routine
    mov rcx, r13                        ; Thread ID as arg
    call pthread_create
    test eax, eax
    jnz .thread_fail
    
    inc r13d
    jmp .create_loop
    
.threads_created:
    mov dword [pool_initialized], 1
    
    ; Print info
    lea rdi, [msg_pool_init]
    mov esi, r12d
    xor eax, eax
    call printf
    
    xor eax, eax
    jmp .done
    
.thread_fail:
    ; Cleanup created threads
    mov dword [shutdown_flag], 1
    lea rdi, [cond_storage]
    call pthread_cond_broadcast
    ; Fall through to cleanup
    
.done_cond_fail:
    lea rdi, [cond_storage]
    call pthread_cond_destroy
    
.cond_fail:
    lea rdi, [mutex_storage]
    call pthread_mutex_destroy
    
.mutex_fail:
    mov eax, -1
    jmp .done
    
.already_init:
    xor eax, eax
    
.done:
    add rsp, 24
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; worker_routine - Thread pool worker function
; Arguments:
;   RDI = thread id
; =============================================================================
worker_routine:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                ; Thread ID
    
.worker_loop:
    ; Lock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_lock
    
    ; Wait for work
.wait_work:
    ; Check shutdown
    cmp dword [shutdown_flag], 1
    je .worker_shutdown
    
    ; Check for work
    cmp qword [queue_count], 0
    jg .has_work
    
    ; Wait on condition
    lea rdi, [cond_storage]
    lea rsi, [mutex_storage]
    call pthread_cond_wait
    jmp .wait_work
    
.has_work:
    ; Dequeue work item
    mov rax, [queue_head]
    lea rbx, [work_queue]
    lea rbx, [rbx + rax*8]      ; Pointer to work item
    
    ; Load work item: func, arg1, arg2, arg3
    mov r13, [rbx]              ; func
    mov r14, [rbx + 8]          ; arg1
    mov r15, [rbx + 16]         ; arg2
    mov rcx, [rbx + 24]         ; arg3
    mov [rsp], rcx              ; Save arg3
    
    ; Update head
    inc rax
    cmp rax, WORK_QUEUE_SIZE * 4
    jl .no_wrap
    xor eax, eax
.no_wrap:
    mov [queue_head], rax
    dec qword [queue_count]
    
    ; Increment active tasks
    inc qword [active_tasks]
    
    ; Unlock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_unlock
    
    ; Execute work
    mov rdi, r14                ; arg1
    mov rsi, r15                ; arg2
    mov rdx, [rsp]              ; arg3
    call r13
    
    ; Lock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_lock
    
    ; Decrement active tasks
    dec qword [active_tasks]
    
    ; Signal if all done
    cmp qword [active_tasks], 0
    jne .not_done
    cmp qword [queue_count], 0
    jne .not_done
    
    lea rdi, [done_cond_storage]
    call pthread_cond_broadcast
    
.not_done:
    ; Unlock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_unlock
    
    jmp .worker_loop
    
.worker_shutdown:
    lea rdi, [mutex_storage]
    call pthread_mutex_unlock
    
    xor eax, eax
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; thread_pool_submit - Submit work to thread pool
; Arguments:
;   RDI = function pointer
;   RSI = arg1
;   RDX = arg2
;   RCX = arg3
; Returns:
;   EAX = 0 on success, -1 if queue full
; =============================================================================
thread_pool_submit:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi                ; func
    mov r13, rsi                ; arg1
    mov r14, rdx                ; arg2
    mov r15, rcx                ; arg3
    
    ; Lock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_lock
    
    ; Check if queue full
    cmp qword [queue_count], WORK_QUEUE_SIZE * 4
    jge .queue_full
    
    ; Enqueue work item
    mov rax, [queue_tail]
    lea rbx, [work_queue]
    lea rbx, [rbx + rax*8]
    
    mov [rbx], r12              ; func
    mov [rbx + 8], r13          ; arg1
    mov [rbx + 16], r14         ; arg2
    mov [rbx + 24], r15         ; arg3
    
    ; Update tail
    add rax, 4
    cmp rax, WORK_QUEUE_SIZE * 4
    jl .no_wrap
    xor eax, eax
.no_wrap:
    mov [queue_tail], rax
    inc qword [queue_count]
    
    ; Signal worker
    lea rdi, [cond_storage]
    call pthread_cond_signal
    
    ; Unlock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_unlock
    
    xor eax, eax
    jmp .done
    
.queue_full:
    lea rdi, [mutex_storage]
    call pthread_mutex_unlock
    mov eax, -1
    
.done:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; thread_pool_wait - Wait for all submitted work to complete
; =============================================================================
thread_pool_wait:
    push rbp
    mov rbp, rsp
    sub rsp, 16
    
    ; Lock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_lock
    
.wait_loop:
    ; Check if all done
    cmp qword [queue_count], 0
    jne .not_done
    cmp qword [active_tasks], 0
    je .all_done
    
.not_done:
    ; Wait on done condition
    lea rdi, [done_cond_storage]
    lea rsi, [mutex_storage]
    call pthread_cond_wait
    jmp .wait_loop
    
.all_done:
    ; Unlock mutex
    lea rdi, [mutex_storage]
    call pthread_mutex_unlock
    
    add rsp, 16
    pop rbp
    ret

; =============================================================================
; thread_pool_shutdown - Shutdown thread pool
; =============================================================================
thread_pool_shutdown:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 16
    
    cmp dword [pool_initialized], 0
    je .not_init
    
    ; Lock and set shutdown flag
    lea rdi, [mutex_storage]
    call pthread_mutex_lock
    
    mov dword [shutdown_flag], 1
    
    ; Wake all workers
    lea rdi, [cond_storage]
    call pthread_cond_broadcast
    
    ; Unlock
    lea rdi, [mutex_storage]
    call pthread_mutex_unlock
    
    ; Join all threads
    xor r12d, r12d
    
.join_loop:
    cmp r12d, [num_workers]
    jge .joined
    
    mov rdi, [worker_threads + r12*8]
    xor esi, esi
    call pthread_join
    
    inc r12d
    jmp .join_loop
    
.joined:
    ; Destroy sync primitives
    lea rdi, [done_cond_storage]
    call pthread_cond_destroy
    
    lea rdi, [cond_storage]
    call pthread_cond_destroy
    
    lea rdi, [mutex_storage]
    call pthread_mutex_destroy
    
    mov dword [pool_initialized], 0
    
    lea rdi, [msg_pool_shutdown]
    xor eax, eax
    call printf
    
.not_init:
    add rsp, 16
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; parallel_for - Execute function in parallel over range
; Arguments:
;   RDI = function pointer (func(start, end, arg))
;   RSI = start index
;   RDX = end index
;   RCX = user argument
; =============================================================================
parallel_for:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                ; func
    mov r13, rsi                ; start
    mov r14, rdx                ; end
    mov r15, rcx                ; arg
    
    ; Ensure pool is initialized
    cmp dword [pool_initialized], 0
    jne .pool_ready
    
    xor edi, edi                ; Auto-detect threads
    call thread_pool_init
    
.pool_ready:
    ; Calculate chunk size
    mov rax, r14
    sub rax, r13                ; total = end - start
    
    xor edx, edx
    mov ecx, [num_workers]
    div rcx                     ; chunk = total / num_workers
    mov rbx, rax                ; chunk size
    
    test rbx, rbx
    jnz .has_chunks
    mov rbx, 1                  ; At least 1 per chunk
    
.has_chunks:
    ; Submit tasks
    mov rcx, r13                ; current = start
    
.submit_loop:
    cmp rcx, r14
    jge .submitted
    
    ; Calculate chunk end
    mov rax, rcx
    add rax, rbx
    cmp rax, r14
    jle .chunk_ok
    mov rax, r14
    
.chunk_ok:
    mov [rsp], rcx              ; Save current
    mov [rsp + 8], rax          ; Save chunk_end
    
    ; Submit task
    mov rdi, r12                ; func
    mov rsi, rcx                ; start
    mov rdx, rax                ; end
    mov rcx, r15                ; arg
    call thread_pool_submit
    
    mov rcx, [rsp + 8]          ; current = chunk_end
    jmp .submit_loop
    
.submitted:
    ; Wait for completion
    call thread_pool_wait
    
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; parallel_map - Apply function to array elements in parallel
; Arguments:
;   RDI = function pointer (func(float) -> float)
;   RSI = float* dst
;   RDX = float* src
;   RCX = count
; =============================================================================
parallel_map:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    
    mov r12, rdi                ; func
    mov r13, rsi                ; dst
    mov r14, rdx                ; src
    mov r15, rcx                ; count
    
    ; Store context for worker
    mov [rsp], r12
    mov [rsp + 8], r13
    mov [rsp + 16], r14
    
    ; Use parallel_for with map_worker
    lea rdi, [map_worker]
    xor esi, esi                ; start = 0
    mov rdx, r15                ; end = count
    lea rcx, [rsp]              ; context
    call parallel_for
    
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; Helper for parallel_map
map_worker:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov r12, rdi                ; start
    mov r13, rsi                ; end
    mov r14, rdx                ; context
    
    ; Load context
    mov r15, [r14]              ; func
    mov rbx, [r14 + 8]          ; dst
    mov rcx, [r14 + 16]         ; src
    
    ; Process range
.map_loop:
    cmp r12, r13
    jge .map_done
    
    ; Call func(src[i])
    movss xmm0, [rcx + r12*4]
    push rcx
    call r15
    pop rcx
    
    ; Store result
    movss [rbx + r12*4], xmm0
    
    inc r12
    jmp .map_loop
    
.map_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
