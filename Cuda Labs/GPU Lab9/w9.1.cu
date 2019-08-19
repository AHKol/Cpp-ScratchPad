 // Workshop 9 - Memory Coalescence
 // w9.1.cu

 #include <iostream>
 #include <cstdlib>
 #include <cuda_runtime.h>
 // to remove intellisense highlighting
 #include <device_launch_parameters.h>
 #ifndef __CUDACC__
 #define __CUDACC__
 #endif
 #include <device_functions.h>

 const int TILE_WIDTH = 16;  // tile width in each direction

 __global__ void matMul(const float* a, const float* b, float* c, int width) { 
     // add kernel code without coalesced access here
 }

 int main(int argc, char* argv[]) {
     // interpret command-line arguments
     if (argc != 2) {
         std::cerr << argv[0] << ": invalid number of arguments\n"; 
         std::cerr << "Usage: " << argv[0] << "  no_of_rows|columns\n"; 
         return 1;
     }
     int n = atoi(argv[1]) * TILE_WIDTH; // number of rows/columns in A, B, C 

     float* d_A;
     float* d_B;
     float* d_C;
     float* h_A = new float[n * n];
     float* h_B = new float[n * n];
     float* h_C = new float[n * n];

     // populate host matrices a and b
     int kk = 0;
     for (int i = 0; i < n; i++)
         for (int j = 0; j < n; j++) {
             h_A[kk] = (float)kk;
             h_B[kk] = (float)kk;
         }

     // calculate the number of blocks
     int nblocks = n / TILE_WIDTH;
     dim3 grid(nblocks, nblocks);
     dim3 threads(TILE_WIDTH, TILE_WIDTH);

     // BLAS Level 3 calculation: C = A * B
     // add code - allocate memory for matrices d_A, d_B, d_C on the device

     // add code - copy h_A and h_B to d_A and d_B (host to device)

     // launch grid of threads
     matMul<<<grid, threads>>>(d_A, d_B, d_C, n);

     // copy C to c (device to host)

     // add code - deallocate d_A, d_B, d_C, h_A, h_B, h_C

     // reset the device
     cudaDeviceReset();
 }