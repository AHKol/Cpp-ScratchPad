// Linear Algebra - Workshop 2
 // w2.custom.cpp

 #include <iostream>
 #include <ctime>
 #include <chrono>
 #include <cstdlib>
 using namespace std::chrono;

 void init(float* a, int n) {
     const float randf = 1.0f / (float) RAND_MAX; 
     for (int i = 0; i < n; i++)
         a[i] = std::rand() * randf;
 }

 void reportTime(const char* msg, steady_clock::duration span) {
     auto ms = duration_cast<milliseconds>(span);
     std::cout << msg << " - took - " <<
      ms.count() << " millisecs" << std::endl;
 }

 float sdot(int n, const float* a, const float* b) {
	 float RET = 0.0;
	 for (int i = 0; i < n; i++) {
		 RET += a[i] * b[i];
	 }
	 return RET;
 }

 void sgemv(const float* a, int n, const float* v, float* w) {

	 for (int i = 0; i < n; i++) {
		 w[i] = 0;
		 for (int j = 0; j < n; j++)
			 w[i] += a[n * i + j] * v[j]; //n * i = row 
	 }
 }

 void sgemm(const float* a, const float* b, int n, float* c) {

	 for (int i = 0; i < n; i++) {
		 for (int j = 0; j < n; j++) {
			 c[n * i +j] = 0; //n * i = row, + j = column
			 for (int k = 0; k < n; k++)
				 c[n * i +j] += a [i * n + k] * b[k * n + j];
		 }
	 }

 }

 int main(int argc, char** argv) {

     // interpret command-line argument
     if (argc != 2) {
         std::cerr << argv[0] << ": invalid number of arguments\n"; 
         std::cerr << "Usage: " << argv[0] << "  size_of_matrices\n"; 
         return 1;
     }
     int n = std::atoi(argv[1]);
     steady_clock::time_point ts, te;
     float* v = new float[n];
     float* w = new float[n];
     float* a = new float[n * n];
     float* b = new float[n * n];
     float* c = new float[n * n];

     // initialization
     std::srand(std::time(nullptr));
     ts = steady_clock::now();
     init(a, n * n);
     init(b, n * n);
     init(v, n);
     init(w, n);
     te = steady_clock::now();
     reportTime("initialization         ", te - ts); 

     // vector-vector - dot product of v and w
     ts = steady_clock::now();
     sdot(n, v, w);
     te = steady_clock::now();
     reportTime("vector-vector operation", te - ts); 

     // matrix-vector - product of a and v
     ts = steady_clock::now();
     sgemv(a, n, v, w);
     te = steady_clock::now();
     reportTime("matrix-vector operation", te - ts); 

     // matrix-matrix - product of a and b
     ts = steady_clock::now();
     sgemm(a, b, n, c);
     te = steady_clock::now();
     reportTime("matrix-matrix operation", te - ts); 

     delete [] v;
     delete [] w;
     delete [] a;
     delete [] b;
     delete [] c;
	 system("pause");
 }