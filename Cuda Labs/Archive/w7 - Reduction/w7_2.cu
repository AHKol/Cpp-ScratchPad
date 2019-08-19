// Reduction - Workshop 7
// Yuriy Kartuzov
// w7_2.cu

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
// CUDA header file
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>

const int ntpb = 1024; // number of threads per block

void init(float* a, int n, bool debug) {
	float f = 1.0f / RAND_MAX;
	for (int i = 0; i < n; i++)
		if (debug)
			a[i] = 1.0f;
		else
			a[i] = std::rand() * f; // [0.0f 1.0f]
}

// kernel 1 - product
__global__ void product(float * a, float * b, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		a[i] = a[i] * b[i];
	}
}

// kernel 2 - reduction on a single block
__global__ void reduction(float * a, float * c ,int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int stride = 1; i + stride < n; stride <<= 1) {
		if (i % (stride * 2) == 0)
			a[i] = a[i] + a[i + stride];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		c[blockIdx.x] = a[i];
}


int main(int argc, char** argv) {
	// interpret command-line arguments
	if (argc != 2 && argc != 3) {
		std::cerr << argv[0] << ": invalid number of arguments\n";
		std::cerr << "Usage: " << argv[0] << "  size_of_vectors\n";
		return 1;
	}
	
	int n = atoi(argv[1]);
	bool debug = argc == 3;
	std::srand((unsigned)time(nullptr));

	// calculate required number of blocks
	int nblks = (n + ntpb -1) / ntpb; 

	// host vectors
	float* h_a = new float[ntpb * nblks];
	float* h_b = new float[ntpb * nblks];
	init(h_a, n, debug);
	init(h_b, n, debug);
	for (int i = n; i < nblks * ntpb; i++) {
		h_a[i] = 0.0f;
		h_b[i] = 0.0f;
	}
	// dot product on the host
	float h_h = 0.f;
	for (int i = 0; i < n; i++)
		h_h += h_a[i] * h_b[i];

	// allocate device vectors (d_a[nblks * ntpb], d_b[n], d_c[nblks])
	float* d_a;
	float* d_b;
	float* d_c;
	cudaMalloc((void **) &d_a, nblks * ntpb * sizeof(float));
	cudaMalloc((void **) &d_b, nblks * ntpb * sizeof(float));
	cudaMalloc((void **) &d_c, nblks        * sizeof(float));

	// copy from the host to the device h_a -> d_a, h_b -> d-b
	cudaMemcpy(d_a, h_a, nblks * ntpb * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nblks * ntpb * sizeof(float), cudaMemcpyHostToDevice);

	// Kerne l - product
	product << <nblks, ntpb >> > (d_a, d_b, n);
	cudaDeviceSynchronize();

	// Kernel 2 - reduction to one value per block
	reduction << <nblks, ntpb >> > (d_a, d_c, n);
	cudaDeviceSynchronize();

	// intermediate debugging output
	if (debug) {
		float* h_c = new float[nblks];
		cudaMemcpy(h_c, d_c, nblks * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < nblks; i++)
			std::cout << h_c[i] << ' ';
		std::cout << std::endl;
		delete[] h_c;
	}

	// reduction of block values to a single value
	reduction << <nblks, ntpb >> > (d_c,d_a, n);
	cudaDeviceSynchronize();

	// copy final result from device to host - from d_c to h_c
	float h_c;
	cudaMemcpy(&h_c,d_a, sizeof(float), cudaMemcpyDeviceToHost);


	// report your results
	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Device = " << h_c << "\nHost   = " << h_h << std::endl;

	// free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// free host memory
	delete[] h_a;
	delete[] h_b;
} 