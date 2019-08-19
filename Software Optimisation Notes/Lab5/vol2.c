#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "vol.h"

//change the array to match new volume level
void setVolume(int16_t lookup[], float volume_factor) {
	for (int i = -32768; i < 32768; i++) {
		lookup[i + 32768] = (int16_t)((volume_factor * (float)i));
	}
}
// Retreve the sample from the lookup based on the input
static inline int16_t scale_sample(int16_t sample, const int16_t lookup[]) {
	return lookup[sample + 32768];
}

int main() {

	// Allocate memory for large in and out arrays
	int16_t*	in;
	int16_t*	out;
	in = (int16_t*)calloc(SAMPLES, sizeof(int16_t));
	out = (int16_t*)calloc(SAMPLES, sizeof(int16_t));

	int	x;
	int	ttl = 0;

	// Seed the pseudo-random number generator
	srand(-1);

	// Fill the array with random data
	for (x = 0; x < SAMPLES; x++) {
		in[x] = (rand() % 65536) - 32768;
	}

	// ######################################
	// This is the interesting part!
	// Scale the volume of all of the samples

	//create an array of posible outputs with a volume of 0.75
	int16_t lookup[65536];  //65536 posible amount of outputs

	setVolume(lookup, 0.75);
	for (x = 0; x < SAMPLES; x++) {
		out[x] = scale_sample(in[x], lookup); //32768 offsets the negative values
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

