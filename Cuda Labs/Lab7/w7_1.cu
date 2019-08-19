// Reduction - Workshop 7
// w7_1.cu

#include <iostream>
#include <cstdlib>
#include <ctime>
// CUDA header file
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>

void init(float* a, int n) {
	float f = 1.0f / RAND_MAX;
	for (int i = 0; i < n; i++)
		a[i] = std::rand() * f; // [0.0f 1.0f]
}

__global__ void dotProduct(float * a, float * b, int n) {
	int i = threadIdx.x;
	a[i] = a[i] * b[i];
	__syncthreads();
	for (int stride = 1; i + stride < n; stride <<= 1) {
		if (i % (2 * stride) == 0)
			a[i] += a[i + stride];
		__syncthreads();
	}
}

int main(int argc, char** argv) {
	// interpret command-line arguments
	if (argc != 2) {
		std::cerr << argv[0] << ": invalid number of arguments\n";
		std::cerr << "Usage: " << argv[0] << "  size_of_vectors\n";
		return 1;
	}
	int n = std::atoi(argv[1]);
	std::srand((unsigned)time(nullptr));

	// host vectors
	float* h_a = new float[n];
	float* h_b = new float[n];
	init(h_a, n);
	init(h_b, n);
	// dot product on the host
	float h_h = 0.f;
	for (int i = 0; i < n; i++)
		h_h += h_a[i] * h_b[i];

	// allocate memory for device vectors (d_a[n], d_b[n])
	float* d_a;
	float* d_b;
	cudaMalloc((void**)&d_a, n * sizeof(float));
	cudaMalloc((void**)&d_b, n * sizeof(float));

	// copy host vectors to device vectors h_a -> d_a, h_b -> d_b
	cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

	// launch the grid of threads
	dotProduct << <1, n >> >(d_a, d_b, n);

	// copy the result from the device to the host d_a -> h_c
	float h_c;
	cudaMemcpy(&h_c, d_a, sizeof(float), cudaMemcpyDeviceToHost);

	// compare the results
	std::cout << "Device = " << h_c << "\nHost   = " << h_h << std::endl;

	// free device memory
	cudaFree(d_a);
	cudaFree(d_b);

	// free host memory
	delete[] h_a;
	delete[] h_b;
}