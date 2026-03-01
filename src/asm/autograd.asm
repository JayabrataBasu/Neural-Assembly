; =============================================================================
; autograd.asm - Automatic Differentiation Engine
; =============================================================================
; Graph node structure, forward ops, backward execution
; =============================================================================

; Node struct layout (64 bytes):
; Offset  Size    Field
; 0       8       value        (Tensor*)
; 8       8       grad         (Tensor*)
; 16      8       backward_fn  (void (*)(Node*))
; 24      4       n_parents    (uint32_t)
; 28      4       visited      (uint32_t)
; 32      8       parents      (Node**)
; 40      8       saved_tensors (void**) - for backward context
; 48      8       n_saved      (uint64_t)
; 56      8       requires_grad (uint64_t) - bits: 0=requires_grad, 1=persistent

%define NODE_SIZE           64
%define NODE_VALUE          0
%define NODE_GRAD           8
%define NODE_BACKWARD_FN    16
%define NODE_N_PARENTS      24
%define NODE_VISITED        28
%define NODE_PARENTS        32
%define NODE_SAVED_TENSORS  40
%define NODE_N_SAVED        48
%define NODE_REQUIRES_GRAD  56

; requires_grad field flags
%define NODE_FLAG_REQUIRES_GRAD  1
%define NODE_FLAG_PERSISTENT     2

; Tensor struct offsets
%define TENSOR_DATA         0
%define TENSOR_NDIM         8
%define TENSOR_SHAPE        16
%define TENSOR_STRIDE       24
%define TENSOR_DTYPE        32

%define DT_FLOAT32          0
%define DT_FLOAT64          1

section .data
    align 8
    err_null_node:      db "Error: Null node in autograd operation", 0
    err_backward_fail:  db "Error: Backward function failed", 0
    one_f32:            dd 1.0
    one_f64:            dq 1.0

section .bss
    align 8
    ; Stack for topological sort (max 1024 nodes)
    topo_stack:         resq 1024
    topo_stack_ptr:     resq 1

section .text

; External functions
extern mem_alloc
extern mem_alloc_aligned
extern mem_free
extern mem_zero
extern mem_copy
extern panic
extern tensor_create
extern tensor_zeros
extern tensor_copy
extern tensor_free
extern tensor_numel
extern tensor_fill
extern tensor_data_size
extern ew_add
extern ew_sub
extern ew_mul
extern ew_div
extern ew_scalar_mul
extern ew_max
extern ew_neg
extern matmul
extern tensor_transpose_2d
extern reduce_sum

; Export autograd functions
global node_create
global node_from_tensor
global node_free
global node_free_graph
global node_free_with_tensor
global zero_grad
global backward
global node_add
global node_sub
global node_mul
global node_matmul
global add_backward
global sub_backward
global mul_backward
global matmul_backward

; =============================================================================
; node_create - Create a new computation graph node
; Arguments:
;   RDI = Tensor* value
;   RSI = flags (bit 0: requires_grad, bit 1: persistent)
; Returns:
;   RAX = Node*
; =============================================================================
node_create:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    mov r12, rdi                    ; value tensor
    mov r13, rsi                    ; flags (requires_grad + persistent)
    
    ; Allocate node struct
    mov rdi, NODE_SIZE
    mov rsi, 16
    call mem_alloc_aligned
    test rax, rax
    jz .alloc_failed
    mov rbx, rax
    
    ; Initialize node
    mov [rbx + NODE_VALUE], r12
    mov qword [rbx + NODE_GRAD], 0
    mov qword [rbx + NODE_BACKWARD_FN], 0
    mov dword [rbx + NODE_N_PARENTS], 0
    mov dword [rbx + NODE_VISITED], 0
    mov qword [rbx + NODE_PARENTS], 0
    mov qword [rbx + NODE_SAVED_TENSORS], 0
    mov qword [rbx + NODE_N_SAVED], 0
    mov [rbx + NODE_REQUIRES_GRAD], r13  ; Store full flags
    
    ; If requires_grad bit is set, allocate gradient tensor
    test r13, NODE_FLAG_REQUIRES_GRAD
    jz .no_grad
    
    ; Create gradient tensor with same shape
    mov rdi, [r12 + TENSOR_NDIM]
    mov rsi, [r12 + TENSOR_SHAPE]
    mov edx, [r12 + TENSOR_DTYPE]
    call tensor_zeros
    mov [rbx + NODE_GRAD], rax

.no_grad:
    mov rax, rbx
    
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_from_tensor - Create a leaf node from tensor
; Arguments:
;   RDI = Tensor* value
;   RSI = requires_grad (0 or 1)
; Returns:
;   RAX = Node*
; =============================================================================
node_from_tensor:
    ; Just calls node_create
    jmp node_create

; =============================================================================
; node_free - Free a node (not its value tensor)
; Arguments:
;   RDI = Node* node
; =============================================================================
node_free:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 8
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi
    
    ; Free gradient tensor if exists
    mov rdi, [r12 + NODE_GRAD]
    test rdi, rdi
    jz .skip_grad
    call tensor_free
.skip_grad:
    
    ; Free parents array if exists
    mov rdi, [r12 + NODE_PARENTS]
    test rdi, rdi
    jz .skip_parents
    call mem_free
.skip_parents:
    
    ; NOTE: Don't free saved_tensors - it may point to persistent data like modules
    ; The saved_tensors field is used as a generic context pointer
    
    ; Free node struct
    mov rdi, r12
    call mem_free

.done:
    add rsp, 8
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_free_with_tensor - Free a node AND its value tensor
; Arguments:
;   RDI = Node* node
; Use this when freeing intermediate nodes that own their tensors
; =============================================================================
node_free_with_tensor:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 8
    
    test rdi, rdi
    jz .nfwt_done
    
    mov r12, rdi
    
    ; Free value tensor first
    mov rdi, [r12 + NODE_VALUE]
    test rdi, rdi
    jz .nfwt_skip_value
    call tensor_free
.nfwt_skip_value:
    
    ; Free gradient tensor if exists
    mov rdi, [r12 + NODE_GRAD]
    test rdi, rdi
    jz .nfwt_skip_grad
    call tensor_free
.nfwt_skip_grad:
    
    ; Free parents array if exists
    mov rdi, [r12 + NODE_PARENTS]
    test rdi, rdi
    jz .nfwt_skip_parents
    call mem_free
.nfwt_skip_parents:
    
    ; NOTE: Don't free saved_tensors - it may point to persistent data like modules
    ; The saved_tensors field is used as a generic context pointer
    
    ; Free node struct
    mov rdi, r12
    call mem_free

.nfwt_done:
    add rsp, 8
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_free_graph - Recursively free an entire computational graph
; Arguments:
;   RDI = Node* root (starting from loss node, traverses to inputs)
; Frees all nodes and their tensors in the computation graph.
; Uses post-order traversal to ensure children are freed before parents.
; =============================================================================
section .bss
    align 8
    ; Visited array for graph freeing (max 1024 nodes)
    free_visited:       resq 1024
    free_visited_count: resq 1

section .text
node_free_graph:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    test rdi, rdi
    jz .nfg_done
    
    mov r12, rdi                    ; root node
    
    ; Reset visited tracking
    mov qword [rel free_visited_count], 0
    
    ; Call recursive helper
    mov rdi, r12
    call node_free_graph_recursive
    
.nfg_done:
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_free_graph_recursive - Internal recursive helper
; Arguments:
;   RDI = Node* node
; =============================================================================
node_free_graph_recursive:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    test rdi, rdi
    jz .nfgr_done
    
    mov r12, rdi                    ; current node
    
    ; Check if node is persistent (model parameters) - don't free those
    mov rax, [r12 + NODE_REQUIRES_GRAD]
    test rax, NODE_FLAG_PERSISTENT
    jnz .nfgr_done                  ; Skip persistent nodes
    
    ; Check if already visited
    mov rcx, [rel free_visited_count]
    test rcx, rcx
    jz .nfgr_not_visited
    
    lea rbx, [rel free_visited]
    xor r8d, r8d
.nfgr_check_visited:
    cmp r8, rcx
    jge .nfgr_not_visited
    cmp [rbx + r8*8], r12
    je .nfgr_done                   ; Already visited, skip
    inc r8
    jmp .nfgr_check_visited
    
.nfgr_not_visited:
    ; Mark as visited
    mov rax, [rel free_visited_count]
    cmp rax, 1024
    jge .nfgr_skip_mark             ; Avoid overflow
    lea rbx, [rel free_visited]
    mov [rbx + rax*8], r12
    inc rax
    mov [rel free_visited_count], rax
.nfgr_skip_mark:
    
    ; First, recursively free parents (children in backward graph)
    mov ecx, [r12 + NODE_N_PARENTS]
    test ecx, ecx
    jz .nfgr_free_self
    
    mov r13, [r12 + NODE_PARENTS]   ; parents array
    xor r14d, r14d                  ; parent index
    
.nfgr_free_parents_loop:
    cmp r14d, [r12 + NODE_N_PARENTS]
    jge .nfgr_free_self
    
    ; Get parent node
    mov rdi, [r13 + r14*8]
    test rdi, rdi
    jz .nfgr_next_parent
    
    ; Recursively free parent
    push r12
    push r13
    push r14
    call node_free_graph_recursive
    pop r14
    pop r13
    pop r12
    
.nfgr_next_parent:
    inc r14d
    jmp .nfgr_free_parents_loop
    
.nfgr_free_self:
    ; Free this node
    ; For leaf nodes with requires_grad=false, DON'T free value tensor (external input)
    ; For leaf nodes with requires_grad=true, free value tensor (computed temporary)
    ; For non-leaf nodes, always free the value tensor (created by operations)
    mov ecx, [r12 + NODE_N_PARENTS]
    test ecx, ecx
    jz .nfgr_check_leaf_ownership
    
    ; Non-leaf node: free with tensor
    mov rdi, r12
    call node_free_with_tensor
    jmp .nfgr_done
    
.nfgr_check_leaf_ownership:
    ; Leaf node - check requires_grad to determine ownership
    mov rax, [r12 + NODE_REQUIRES_GRAD]
    test rax, rax
    jz .nfgr_free_leaf_external
    
    ; Leaf with requires_grad=true owns its tensor (computed value like W^T)
    mov rdi, r12
    call node_free_with_tensor
    jmp .nfgr_done
    
.nfgr_free_leaf_external:
    ; Leaf with requires_grad=false: external input, don't free tensor
    mov rdi, r12
    call node_free
    
.nfgr_done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; zero_grad - Zero all gradients in the graph reachable from node
; Arguments:
;   RDI = Node* root
; =============================================================================
zero_grad:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi
    
    ; Zero this node's gradient
    mov rdi, [r12 + NODE_GRAD]
    test rdi, rdi
    jz .no_grad
    
    push r12
    mov rdi, [r12 + NODE_GRAD]
    call tensor_data_size
    mov rsi, rax
    mov rdi, [r12 + NODE_GRAD]
    mov rdi, [rdi + TENSOR_DATA]
    call mem_zero
    pop r12

.no_grad:
    ; Recursively zero parents
    mov ecx, [r12 + NODE_N_PARENTS]
    test ecx, ecx
    jz .done
    
    mov r13, [r12 + NODE_PARENTS]
    xor ebx, ebx

.parent_loop:
    cmp ebx, [r12 + NODE_N_PARENTS]
    jge .done
    
    mov rdi, [r13 + rbx*8]
    push rbx
    push r12
    push r13
    call zero_grad
    pop r13
    pop r12
    pop rbx
    
    inc ebx
    jmp .parent_loop

.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; backward_topo_sort - Build reverse topological order
; Arguments:
;   RDI = Node* node
; Visits nodes and pushes to topo_stack in reverse order
; =============================================================================
backward_topo_sort:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 8
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi
    
    ; Check if already visited
    mov eax, [r12 + NODE_VISITED]
    test eax, eax
    jnz .done
    
    ; Mark as visited
    mov dword [r12 + NODE_VISITED], 1
    
    ; Visit parents first
    mov ecx, [r12 + NODE_N_PARENTS]
    test ecx, ecx
    jz .push_self
    
    mov r13, [r12 + NODE_PARENTS]
    xor ebx, ebx

.visit_parents:
    cmp ebx, [r12 + NODE_N_PARENTS]
    jge .push_self
    
    mov rdi, [r13 + rbx*8]
    push rbx
    push r12
    push r13
    call backward_topo_sort
    pop r13
    pop r12
    pop rbx
    
    inc ebx
    jmp .visit_parents

.push_self:
    ; Push this node to stack
    mov rax, [rel topo_stack_ptr]
    lea rbx, [rel topo_stack]
    mov [rbx + rax*8], r12
    inc rax
    mov [rel topo_stack_ptr], rax

.done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; backward - Run backward pass from loss node
; Arguments:
;   RDI = Node* loss_node
; =============================================================================
backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 16
    
    test rdi, rdi
    jz .done
    
    mov r12, rdi                    ; loss_node
    
    ; Initialize loss gradient to 1.0
    mov rdi, [r12 + NODE_GRAD]
    test rdi, rdi
    jz .create_grad
    
    ; Fill with 1.0
    mov rax, 1
    cvtsi2sd xmm0, rax              ; 1.0 as double
    mov rdi, [r12 + NODE_GRAD]
    call tensor_fill
    jmp .do_topo_sort

.create_grad:
    ; Create gradient tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_zeros
    mov [r12 + NODE_GRAD], rax
    
    ; Fill with 1.0
    mov rax, 1
    cvtsi2sd xmm0, rax
    mov rdi, [r12 + NODE_GRAD]
    call tensor_fill

.do_topo_sort:
    ; Reset stack pointer
    mov qword [rel topo_stack_ptr], 0
    
    ; Build topological order
    mov rdi, r12
    call backward_topo_sort
    
    ; Process nodes in reverse order (pop from stack)
    mov r13, [rel topo_stack_ptr]   ; stack size
    
.backward_loop:
    test r13, r13
    jz .clear_visited
    
    dec r13
    lea rbx, [rel topo_stack]
    mov r14, [rbx + r13*8]  ; current node
    
    ; Call backward function if exists
    mov rax, [r14 + NODE_BACKWARD_FN]
    test rax, rax
    jz .backward_loop
    
    ; Call backward_fn(node)
    mov rdi, r14
    call rax
    
    jmp .backward_loop

.clear_visited:
    ; Reset visited flags
    mov r13, [rel topo_stack_ptr]
.clear_loop:
    test r13, r13
    jz .done
    dec r13
    lea rbx, [rel topo_stack]
    mov rax, [rbx + r13*8]
    mov dword [rax + NODE_VISITED], 0
    jmp .clear_loop

.done:
    add rsp, 16
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_add - Addition forward: out = a + b
; Arguments:
;   RDI = Node* a
;   RSI = Node* b
; Returns:
;   RAX = Node* out
; =============================================================================
node_add:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; a
    mov r13, rsi                    ; b
    
    ; Create output tensor
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r14, rax                    ; output tensor
    
    ; Compute forward: out = a + b
    mov rdi, r14
    mov rsi, [r12 + NODE_VALUE]
    mov rdx, [r13 + NODE_VALUE]
    call ew_add
    
    ; Create output node
    mov rdi, r14
    mov rsi, 1                      ; requires_grad
    call node_create
    test rax, rax
    jz .alloc_failed
    mov r15, rax                    ; output node
    
    ; Set backward function
    lea rax, [rel add_backward]
    mov [r15 + NODE_BACKWARD_FN], rax
    
    ; Set parents
    mov dword [r15 + NODE_N_PARENTS], 2
    mov rdi, 16                     ; 2 pointers
    call mem_alloc
    mov [r15 + NODE_PARENTS], rax
    mov [rel rax], r12                  ; parent 0 = a
    mov [rax + 8], r13              ; parent 1 = b
    
    mov rax, r15
    
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
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
; add_backward - Backward for addition
; Arguments:
;   RDI = Node* self
; =============================================================================
add_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 8
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]   ; parents array
    
    ; dL/da += dL/dout
    mov rax, [rel r14]                  ; parent a
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .skip_a
    
    mov rsi, [rax + NODE_GRAD]      ; current grad
    mov rdx, r13                    ; dL/dout
    call ew_add                     ; grad_a += dL/dout

.skip_a:
    ; dL/db += dL/dout
    mov rax, [r14 + 8]              ; parent b
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .done
    
    mov rsi, [rax + NODE_GRAD]
    mov rdx, r13
    call ew_add

.done:
    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_sub - Subtraction forward: out = a - b
; =============================================================================
node_sub:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi
    mov r13, rsi
    
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r14, rax
    
    mov rdi, r14
    mov rsi, [r12 + NODE_VALUE]
    mov rdx, [r13 + NODE_VALUE]
    call ew_sub
    
    mov rdi, r14
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov r15, rax
    
    lea rax, [rel sub_backward]
    mov [r15 + NODE_BACKWARD_FN], rax
    
    mov dword [r15 + NODE_N_PARENTS], 2
    mov rdi, 16
    call mem_alloc
    mov [r15 + NODE_PARENTS], rax
    mov [rel rax], r12
    mov [rax + 8], r13
    
    mov rax, r15
    
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
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
; sub_backward - Backward for subtraction
; =============================================================================
sub_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    mov r12, rdi
    mov r13, [r12 + NODE_GRAD]
    mov r14, [r12 + NODE_PARENTS]
    
    ; dL/da += dL/dout
    mov rax, [rel r14]
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .skip_a
    
    mov rsi, [rax + NODE_GRAD]
    mov rdx, r13
    call ew_add

.skip_a:
    ; dL/db -= dL/dout (or += -dL/dout)
    mov rax, [r14 + 8]
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .done
    
    ; Create temp for negated gradient
    mov rax, [r14 + 8]
    mov r15, rax
    mov rax, [rax + NODE_GRAD]
    push rax
    
    mov rax, r13                    ; dL/dout tensor
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov rbx, rax                    ; temp tensor
    
    ; Negate: temp = -dL/dout
    mov rdi, rbx
    mov rsi, r13
    call ew_neg
    
    ; grad_b += temp
    pop rdi
    mov rsi, rdi
    mov rdx, rbx
    call ew_add
    
    ; Free temp
    mov rdi, rbx
    call tensor_free

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
; node_mul - Elementwise multiplication: out = a * b
; =============================================================================
node_mul:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi
    mov r13, rsi
    
    mov rax, [r12 + NODE_VALUE]
    mov rdi, [rax + TENSOR_NDIM]
    mov rsi, [rax + TENSOR_SHAPE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov r14, rax
    
    mov rdi, r14
    mov rsi, [r12 + NODE_VALUE]
    mov rdx, [r13 + NODE_VALUE]
    call ew_mul
    
    mov rdi, r14
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov r15, rax
    
    lea rax, [rel mul_backward]
    mov [r15 + NODE_BACKWARD_FN], rax
    
    mov dword [r15 + NODE_N_PARENTS], 2
    mov rdi, 16
    call mem_alloc
    mov [r15 + NODE_PARENTS], rax
    mov [rel rax], r12
    mov [rax + 8], r13
    
    mov rax, r15
    
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
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
; mul_backward - Backward for elementwise multiplication
; dL/da = dL/dout * b
; dL/db = dL/dout * a
; =============================================================================
mul_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout
    mov r14, [r12 + NODE_PARENTS]   ; parents
    
    mov rax, [rel r14]                  ; parent a
    mov rbx, [r14 + 8]              ; parent b
    
    ; Create temp tensor for products
    mov rcx, r13
    mov rdi, [rcx + TENSOR_NDIM]
    mov rsi, [rcx + TENSOR_SHAPE]
    mov edx, [rcx + TENSOR_DTYPE]
    call tensor_create
    mov r15, rax                    ; temp tensor
    
    ; dL/da += dL/dout * b
    mov rax, [rel r14]
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .skip_a
    
    ; temp = dL/dout * b
    mov rdi, r15
    mov rsi, r13
    mov rdx, [rbx + NODE_VALUE]
    call ew_mul
    
    ; grad_a += temp
    mov rax, [rel r14]
    mov rdi, [rax + NODE_GRAD]
    mov rsi, rdi
    mov rdx, r15
    call ew_add

.skip_a:
    ; dL/db += dL/dout * a
    mov rax, [r14 + 8]
    mov rdi, [rax + NODE_GRAD]
    test rdi, rdi
    jz .cleanup
    
    mov rax, [rel r14]
    ; temp = dL/dout * a
    mov rdi, r15
    mov rsi, r13
    mov rdx, [rax + NODE_VALUE]
    call ew_mul
    
    ; grad_b += temp
    mov rax, [r14 + 8]
    mov rdi, [rax + NODE_GRAD]
    mov rsi, rdi
    mov rdx, r15
    call ew_add

.cleanup:
    mov rdi, r15
    call tensor_free

    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; node_matmul - Matrix multiplication: out = A @ B
; Arguments:
;   RDI = Node* A (M x K)
;   RSI = Node* B (K x N)
; Returns:
;   RAX = Node* out (M x N)
; =============================================================================
node_matmul:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    
    mov r12, rdi                    ; A
    mov r13, rsi                    ; B
    
    ; Get dimensions
    mov rax, [r12 + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov r14, [rel rax]                  ; M
    mov rbx, [rax + 8]              ; K
    
    mov rax, [r13 + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov r15, [rax + 8]              ; N
    
    ; Create output tensor (M x N)
    mov qword [rel rsp], r14            ; shape[0] = M
    mov qword [rsp+8], r15          ; shape[1] = N
    
    mov rdi, 2                      ; ndim = 2
    lea rsi, [rel rsp]                  ; shape
    mov rax, [r12 + NODE_VALUE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    test rax, rax
    jz .alloc_failed
    mov [rsp+16], rax               ; output tensor
    
    ; Compute matmul
    mov rdi, rax
    mov rsi, [r12 + NODE_VALUE]
    mov rdx, [r13 + NODE_VALUE]
    call matmul
    
    ; Create output node
    mov rdi, [rsp+16]
    mov rsi, 1
    call node_create
    test rax, rax
    jz .alloc_failed
    mov [rsp+24], rax               ; output node
    
    ; Set backward function
    lea rcx, [rel matmul_backward]
    mov [rax + NODE_BACKWARD_FN], rcx
    
    ; Set parents
    mov dword [rax + NODE_N_PARENTS], 2
    mov rdi, 16
    push rax
    call mem_alloc
    pop rcx
    mov [rcx + NODE_PARENTS], rax
    mov [rel rax], r12
    mov [rax + 8], r13
    
    mov rax, [rsp+24]
    
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_failed:
    xor eax, eax
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; =============================================================================
; matmul_backward - Backward for matrix multiplication
; out = A @ B
; dL/dA = dL/dout @ B^T
; dL/dB = A^T @ dL/dout
; =============================================================================
matmul_backward:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56
    
    mov r12, rdi                    ; self
    mov r13, [r12 + NODE_GRAD]      ; dL/dout (M x N)
    mov r14, [r12 + NODE_PARENTS]
    
    mov rbx, [rel r14]                  ; parent A (M x K)
    mov r15, [r14 + 8]              ; parent B (K x N)
    
    ; Get dimensions
    mov rax, [rbx + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov rcx, [rel rax]                  ; M
    mov [rel rsp], rcx
    mov rcx, [rax + 8]              ; K
    mov [rsp+8], rcx
    
    mov rax, [r15 + NODE_VALUE]
    mov rax, [rax + TENSOR_SHAPE]
    mov rcx, [rax + 8]              ; N
    mov [rsp+16], rcx
    
    ; dL/dA = dL/dout @ B^T
    mov rax, [rbx + NODE_GRAD]
    test rax, rax
    jz .skip_grad_a
    
    ; Create B^T (N x K)
    mov rdi, [rsp+16]               ; N
    mov [rsp+24], rdi
    mov rdi, [rsp+8]                ; K
    mov [rsp+32], rdi
    
    mov rdi, 2
    lea rsi, [rsp+24]               ; shape [N, K]
    mov rax, [r15 + NODE_VALUE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+40], rax               ; B^T tensor
    
    ; Transpose B into B^T
    mov rdi, rax
    mov rsi, [r15 + NODE_VALUE]
    call tensor_transpose_2d
    
    ; Create temp for dL/dout @ B^T (M x K)
    mov rdi, [rel rsp]                  ; M
    mov [rsp+24], rdi
    mov rdi, [rsp+8]                ; K
    mov [rsp+32], rdi
    
    mov rdi, 2
    lea rsi, [rsp+24]
    mov rax, [rbx + NODE_VALUE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+48], rax               ; temp tensor
    
    ; temp = dL/dout @ B^T
    mov rdi, rax
    mov rsi, r13
    mov rdx, [rsp+40]
    call matmul
    
    ; grad_A += temp
    mov rdi, [rbx + NODE_GRAD]
    mov rsi, rdi
    mov rdx, [rsp+48]
    call ew_add
    
    ; Free temps
    mov rdi, [rsp+40]
    call tensor_free
    mov rdi, [rsp+48]
    call tensor_free

.skip_grad_a:
    ; dL/dB = A^T @ dL/dout
    mov rax, [r15 + NODE_GRAD]
    test rax, rax
    jz .done
    
    ; Create A^T (K x M)
    mov rdi, [rsp+8]                ; K
    mov [rsp+24], rdi
    mov rdi, [rel rsp]                  ; M
    mov [rsp+32], rdi
    
    mov rdi, 2
    lea rsi, [rsp+24]
    mov rax, [rbx + NODE_VALUE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+40], rax               ; A^T tensor
    
    ; Transpose A
    mov rdi, rax
    mov rsi, [rbx + NODE_VALUE]
    call tensor_transpose_2d
    
    ; Create temp for A^T @ dL/dout (K x N)
    mov rdi, [rsp+8]                ; K
    mov [rsp+24], rdi
    mov rdi, [rsp+16]               ; N
    mov [rsp+32], rdi
    
    mov rdi, 2
    lea rsi, [rsp+24]
    mov rax, [r15 + NODE_VALUE]
    mov edx, [rax + TENSOR_DTYPE]
    call tensor_create
    mov [rsp+48], rax
    
    ; temp = A^T @ dL/dout
    mov rdi, rax
    mov rsi, [rsp+40]
    mov rdx, r13
    call matmul
    
    ; grad_B += temp
    mov rdi, [r15 + NODE_GRAD]
    mov rsi, rdi
    mov rdx, [rsp+48]
    call ew_add
    
    ; Free temps
    mov rdi, [rsp+40]
    call tensor_free
    mov rdi, [rsp+48]
    call tensor_free

.done:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
