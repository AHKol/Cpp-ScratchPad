.text
.globl    _start

start = 0                       /* starting value for the loop index; note that this is a symbol (constant), not a variable */
max = 15                        /* loop exits when the index hits this number (loop condition is i<max) */

_start:
  mov     $start,%r15         /* loop index */

loop:
/*add first digit to message*/
  mov     %r15,%rax           /*ready division*/
  mov     $0,%rdx
  mov     $10,%r12
  div     %r12                /*divide by tens*/
  mov     %rdx,%r13           /*copy first digit to r13*/
  add     $'0',%r13           /*r13 = acii number of index*/
  mov     %r13b,msg+7         /*Print symbol into memory`*/
/*test if there is a tenth digit*/
  mov     %r15,%rax           /*ready division*/
  mov     $0,%rdx
  mov     $10,%r12
  div     %r12                /*divide by tens*/
  mov     %rax,%r13           /*copy digit r13*/
  cmp     $0,%r13             /*if tens digit doesnt exist*/
  je      skipaddtens         /*skip adding character to message*/

/*add tens character*/
  add     $'0',%r13           /*turn r13 into char of digit*/
  mov     %r13b,msg+6         /*add digit to message*/

skipaddtens:
  mov     $len,%rdx           /*message length*/
  mov     $msg,%rsi           /*message location*/
  mov     $1,%rax             /* stdout for syscall  */
  syscall

  inc     %r15                /* increment index */
  cmp     $max,%r15           /* see if we're done */
  jne     loop                /* loop if we're not */

  mov     $0,%rdi             /* exit status */
  mov     $60,%rax            /* syscall sys_exit */
  syscall

.data                           /*Prevent segmentation fault*/
msg: .ascii     "Loop :  \n"
.set len, . - msg
