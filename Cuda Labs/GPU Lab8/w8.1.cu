// Thread Divergence - Workshop 8
 // w8.1.cu

 #include <iostream>
 #include <cstdlib>
 #include <cuda_runtime.h>
 // to remove intellisense highlighting
 #include <device_launch_parameters.h>
 #ifndef __CUDACC__
 #define __CUDACC__
 #endif
 #include <device_functions.h>

 const int ntpb = 1024; // number of threads per block

 void init(float* a, int n) {
     float f = 1.0f / RAND_MAX;
     for (int i = 0; i < n; i++)
         a[i] = std::rand() * f; // [0.0f 1.0f]
 }

 // calculate the dot product block by block
 __global__ void dotProduct(const float* a, const float* b, float* c, int n) {
     // store the product of a[i] and b[i] in shared memory
     // sum the data in shared memory
     // store the sum in c[blockIdx.x]
 }

 // accumulate the block sums
 __global__ void accumulate(float* c, int n) {
     // store the elements of c[] in shared memory
     // sum the data in shared memory
     // store the sum in c[0]
 }

 int main(int argc, char** argv) {
     // interpret command-line arguments
     if (argc != 2) {
         std::cerr << argv[0] << ": invalid number of arguments\n"; 
         std::cerr << "Usage: " << argv[0] << "  size_of_vectors\n"; 
         return 1;
     }
     int n = std::atoi(argv[1]);
     int nblocks = (n + ntpb - 1) / ntpb;
     if (nblocks > ntpb) {
         nblocks = ntpb;
         n = nblocks * ntpb;
     }

     // host vectors
     float* h_a = new float[n];
     float* h_b = new float[n];
     init(h_a, n);
     init(h_b, n);
     // device vectors (d_a, d_b, d_c)
     float* d_a;
     float* d_b;
     float* d_c;
     cudaMalloc((void**)&d_a, n * sizeof(float));
     cudaMalloc((void**)&d_b, n * sizeof(float));
     cudaMalloc((void**)&d_c, nblocks * sizeof(float));

     // copy from host to device h_a -> d_a, h_b -> d_b
     cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

     // dot product on the device
     dotProduct<<<nblocks, ntpb>>>(d_a, d_b, d_c, n);

     // synchronize
     cudaDeviceSynchronize();

     // accumulate the block sums on the device
     accumulate<<< 1, nblocks>>>(d_c, nblocks);

     // copy from device to host d_c[0] -> h_d
     float h_c;
     cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost); 

     float hx = 0.f;
     for (int i = 0; i < n; i++)
         hx += h_a[i] * h_b[i];
     // compare results
     std::cout << "Device = " << h_c << " Host = " << hx << std::endl; 

     // free device memory
     cudaFree(d_a);
     cudaFree(d_b);
     cudaFree(d_c);

     // free host memory
     delete [] h_a;
     delete [] h_b;

     // reset the device
     cudaDeviceReset();
 }