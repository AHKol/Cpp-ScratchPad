#include "cpuLU.h"
#include "solveLU.h"
#include <iostream>

using namespace std;
int main() {
	int n = 2;

	//double* A = new double;
	double* L = new double[n * n];
	double* U = new double[n * n];
	//double* B = new double;
	double* X = new double[n];
	

	//static test values
	
	double A[4]{ 1, 2, 3, 4 };
	double B[2]{ 1, 1 };

	LU(A, L, U, n);

	solveLU(L, U, n, B, X);

	cout << "Test Done" << endl;
}