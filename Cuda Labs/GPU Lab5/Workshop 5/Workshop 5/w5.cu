// Vector Magnitude - Workshop 5
// w5.cu

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <cstdlib>
#include <algorithm>
// insert thrust header files here
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\generate.h>
#include <thrust\functional.h>
#include <thrust\copy.h>
using namespace std::chrono;

// report system time
void reportTime(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}

// Square Function Object - add class definition here
class square {
public:
	__host__ __device__ float operator()(float input){
		return input * input;
	}
};

// magnitude - add calculation steps here
float magnitude(thrust::host_vector<float>* h_v, int n) {
	float result;

	thrust::device_vector<float> d_v = *h_v;

	//Two Part Version
	/*
	thrust::transform(d_v.begin(), d_v.end(), d_v.begin(), square());
	result = thrust::reduce(d_v.begin(), d_v.end());
	*/
	
	//fused Version
	thrust::transform_reduce(d_v.begin(), d_v.end(), square(), 0.0, thrust::plus<float>())
	
	return result;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << argv[0] << ": invalid number of arguments\n";
		std::cerr << "Usage: " << argv[0] << "  size_of_vector\n";
		return 1;
	}
	int n = std::atoi(argv[1]); // number of elements
	steady_clock::time_point ts, te;

	// Thrust definition of host vector
	thrust::host_vector<float> h_v(n);

	// initialize the host vector
	ts = steady_clock::now();
	std::generate(h_v.begin(), h_v.end(), std::rand);
	te = steady_clock::now();
	reportTime("initialization", te - ts);

	// calculate the magnitude of the host vector
	ts = steady_clock::now();
	float len = magnitude(&h_v, n);
	te = steady_clock::now();
	reportTime("magnitude calculation", te - ts);

	// display the magnitude
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "Magnitude : " << len << std::endl;
}