// Level 3 cuBLAS - Workshop 4
 // w4_cublas.cu
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std::chrono;

// indexing function (column major order)
//
inline int idx(int r, int c, int n)
{
	// ... add indexing formula
	return (c * n) + r;
}

// display matrix M, which is stored in column-major order
//
void display(const char* str, const float* M, int nr, int nc)
{
	std::cout << str << std::endl;
	std::cout << std::fixed << std::setprecision(4);
	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++)
			std::cout << std::setw(10) << M[idx(i, j, nr)];// ... access in column-major order;
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// report system time
//
void reportTime(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}


// matrix multiply
//
void sgemm(const float* h_A, const float* h_B, float* h_C, const int n) {
	steady_clock::time_point ts, te;

	// level 3 calculation: C = alpha * A * B + beta * C

	ts = steady_clock::now();
	// ... allocate memory on the device
	float* d_a;
	float* d_b;
	float* d_c;
	cudaMalloc((void**)&d_a, (n * n) * sizeof(float));
	cudaMalloc((void**)&d_b, (n * n) * sizeof(float));
	cudaMalloc((void**)&d_c, (n * n) * sizeof(float));
	te = steady_clock::now();
	reportTime("allocation of device memory for matrices d_A, d_B and d_C", te - ts);

	// ... create cuBLAS context
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	ts = steady_clock::now();

	// ... copy host matrices to the device
	cublasSetMatrix(n, n, sizeof(float), h_A, n, d_a, n);
	cublasSetMatrix(n, n, sizeof(float), h_B, n, d_b, n);
	te = steady_clock::now();
	reportTime("copying of matrices h_A and h_B to device memory", te - ts);

	ts = steady_clock::now();

	// ... calculate matrix-matrix product
	float alpha = 1.0f;
	float beta = 1.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
	te = steady_clock::now();
	cublasGetMatrix(n, n, sizeof(float), d_c, n, h_C, n);
	reportTime("matrix-matrix multiplication", te - ts);

	// ... copy result matrix from the device to the host
	cudaMemcpy(h_C, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
	te = steady_clock::now();
	reportTime("copying of matrix d_C from device", te - ts);

	// ... destroy cuBLAS context

	ts = steady_clock::now();

	// ... deallocate device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	te = steady_clock::now();
	reportTime("deallocation of device memory for matrices A, B and C", te - ts);

}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cerr << argv[0] << ": invalid number of arguments\n";
		std::cerr << "Usage: " << argv[0] << "  size_of_matrices\n";
		return 1;
	}
	int n = std::atoi(argv[1]); // no of rows/columns in A, B, C 

	// allocate host memory
	float* h_A = new float[n * n];
	float* h_B = new float[n * n];
	float* h_C = new float[n * n];

	// populate host matrices a and b
	for (int i = 0, kk = 0; i < n; i++)
		for (int j = 0; j < n; j++, kk++)
			h_A[kk] = h_B[kk] = (float)kk;

	// C = A * B
	sgemm(h_A, h_B, h_C, n);

	// display results
	if (n <= 5) {
		display("A :", h_A, n, n);
		display("B :", h_B, n, n);
		display("C = A B :", h_C, n, n);
	}

	// deallocate host memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
}
