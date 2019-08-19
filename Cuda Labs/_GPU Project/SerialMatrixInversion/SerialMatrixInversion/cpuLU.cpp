#include "cpuLU.h"

#include <iostream>

void LU(double ** A, double ** L, double** U, int n)
{
	//pivot variable
	double pivot;

	//for every column
	for (int col = 0; col < n; col++) {
		//for every row in lower half
		for (int row = col + 1; row < n; row++) {
			//get pivot 
			pivot = temp[row * n + col] / temp[(row - 1) * n + col];
			//apply pivot to element and everything right of element
			for (int i = col; i < n; i++) {
				temp[row * n + i] -= pivot;
			}
			//apply pivot to lower table
			ret[row * n + col] = pivot;
		}
	}

	//output U
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << temp[i * n + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	//output L
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << ret[i * n + j] << " ";
		}
		std::cout << std::endl;
	}

	return;
}
