.text
.globl    _start

start = 0                       /* starting value for the loop index; note that this is a symbol (constant), not a variable */
max = 15                        /* loop exits when the index hits this number (loop condition is i<max) */

_start:
  mov     x19,start         /*r15 is loop index */
//========================================
 loop:
 /*add first digit to message*/
   /*find quotent then find remander*/
   mov     x14,10            /*ready division*/
   udiv    x13,x19,x14       /*r13 is 10th digit*/
   msub    x12,x19,x13,x14   /*r12 is index mod 10*/
   /*msub r0,r1,r2,r3  // load r0 with r3-(r1*r2) */

   add     x13,x13,48        /*r13 = acii character of index value*/
   mov     x14,msg           /*temporary register for memory poiter*/
   strb    w13,[x14,7]       /*Print first digit char into memory*/
 /*test if there is a tenth digit*/
   mov     x20,0            /*make a zero to compare*/
  cmp     x13,x20             /*if tens digit doesnt exist*/
   b.eq    skipaddtens       /*skip adding character to message*/

 /*add tens character*/
   add     x13,x13,48        /*turn r13 into char of digit*/
   strb    w13,[x14,6]       /*add digit to message*/
   
//===================================================
skipaddtens:
  mov     x2,len            /*message length*/
  mov     x1,msg            /*message location*/
  mov     x0,1              /* stdout for syscall  */
  mov     x8,64             /*write syscall*/
  svc     0

/* Check and loop */
  add     x19,x19,1         /* increment index */
  cmp     x19,max           /* see if we're done */
  b.ne    loop              /* loop if we're not */

/* Exit */
  mov     x0, 0             /* exit status */
  mov     x8, 93            /* syscall sys_exit */
  svc     0                 /* invoke syscall */

.data                       /*Prevent segmentation fault*/
msg: .ascii     "Loop:   \n"
len = . - msg
