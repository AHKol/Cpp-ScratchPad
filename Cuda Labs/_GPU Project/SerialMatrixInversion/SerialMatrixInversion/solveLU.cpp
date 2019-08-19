#include "solveLU.h"

void solveLU(double * L, double * U, int n, double * B, double * X)
{
	//LUx=b
	//let Ux = y
	//solve Ly=b

	double sum;
	double* y = new double[n];
	//Ux = y
	for (int i = 0; i < n; i++) {
		sum = 0;
		for (int j = 0; j < n; j++)
			sum += U[i * n + j] * X[j];
		y[i] = sum;
	}

	//Ly = x
	for (int i = 0; i < n; i++) {
		sum = 0;
		for (int j = 0; j < n; j++)
			sum += L[i * n + j] * y[j];
		X[i] = sum;
	}
}
