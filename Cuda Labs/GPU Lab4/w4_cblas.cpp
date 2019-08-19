 // Level 3 cblas - Workshop 4
 // w4_cblas.cpp

 #include <iostream>
 #include <iomanip>
 #include <cstdlib>
 #include <chrono>
 using namespace std::chrono;

 // indexing function (column major order)
 //
 inline int idx(int r, int c, int n)
 {
     // ... add indexing formula
	 return c * n + r;
 }

 // display matrix M, which is stored in column-major order
 //
 void display(const char* str, const float* M, int nr, int nc)
 {
     std::cout << str << std::endl;
     std::cout << std::fixed << std::setprecision(4);
     for (int i = 0; i < nr; i++) {
         for (int j = 0; j < nc; j++)
             std::cout << std::setw(10)
              << idx(i, j, nr);// ... access in column-major order;
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
 void sgemm(const float* A, const float* B, float* C, int n) {
     steady_clock::time_point ts, te;

     // level 3 calculation: C = alpha * A * B + beta * C
     // add any preliminaries

     ts = steady_clock::now();

     // ... add call to cblas sgemm
	 //cblas_sgemm(n, n, n, n, n, 1, A, n, B, n, 1, C, n);
    

	 for (int i = 0; i < n; i++) {
		 for (int j = 0; j < n; j++) {
			 C[n * i + j] = 0; //n * i = row, + j = column
			 for (int k = 0; k < n; k++)
				 C[n * i + j] += A[i * n + k] * B[k * n + j];
		 }
	 }
	 
	 
	 te = steady_clock::now();
     reportTime("matrix-matrix multiplication", te - ts);
     
 }

 int main(int argc, char* argv[]) {
     if (argc != 2) {
         std::cerr << argv[0] << ": invalid number of arguments\n"; 
         std::cerr << "Usage: " << argv[0] << "  size_of_matrices\n"; 
		 system("pause");
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
     delete [] h_A;
     delete [] h_B;
     delete [] h_C;

	 system("pause");
 }