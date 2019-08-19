
// from GPU610 w7 - Reduction
void init(float* a, int n) {
    float f = 1.0f / RAND_MAX;
    for (int i = 0; i < n; i++)
        a[i] = std::rand() * f; // [0.0f 1.0f]
}