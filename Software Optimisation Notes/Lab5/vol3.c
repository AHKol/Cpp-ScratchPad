#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "vol.h"

int main() {

	// Allocate memory for large in and out arrays
	int16_t*	in;
	int16_t*	out;
	in = (int16_t*)calloc(SAMPLES, sizeof(int16_t));
	out = (int16_t*)calloc(SAMPLES, sizeof(int16_t));

	int x;
	int ttl = 0;

	// Seed the pseudo-random number generator
	srand(-1);

	// Fill the array with random data
	for (x = 0; x < SAMPLES; x++) {
		in[x] = (rand() % 65536) - 32768;
	}

	// ######################################
	// This is the interesting part!
	// Scale the volume of all of the samples

	//set volume factor to int
	int16_t mul = 0b100000000 * 0.75;

	//perform calculation
	for (x = 0; x < SAMPLES; x++) {
		out[x] = (in[x] * mul) >> 8;
	}

	// ######################################

	// Sum up the data
	for (x = 0; x < SAMPLES; x++) {
		ttl = (ttl + out[x]) % 1000;
	}

	// Print the sum
	printf("Result: %d\n", ttl);
	for (int i = 0; i < 5; i++) {
		printf("%d\n", out[i]);
	}
	return 0;

}
