#include "QR.h"
using namespace std;
using namespace chrono;

double QR(int m, bool debug) {

	steady_clock::time_point timeStart, timeEnd;// variables for timing
	cusolverDnHandle_t cusolverH; // cusolver handle
	cublasHandle_t cublasH; // cublas handle
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	// variables for error checking in cudaMalloc
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	//const int m = 5000; // number of rows of A
	const int lda = m; // leading dimension of A
	const int ldb = m; // leading dimension of B
	const int nrhs = 1; // number of right hand sides
						// A - mxm coeff . matr ., B=A*B1 -right hand side , B1 - mxnrhs
	double *A, *B, *B1, *X; // - auxil .matrix , X - solution
							// prepare memory on the host
	A = (double *)malloc(lda*m * sizeof(double));
	B = (double *)malloc(ldb* nrhs * sizeof(double));
	B1 = (double *)malloc(ldb * nrhs * sizeof(double));
	X = (double *)malloc(ldb* nrhs * sizeof(double));
	for (int i = 0; i<lda*m; i++) A[i] = rand() / (double)RAND_MAX;
	for (int i = 0; i<ldb* nrhs; i++) B[i] = 0.0;;
	for (int i = 0; i<ldb* nrhs; i++) B1[i] = 1.0;
	double al = 1.0, bet = 0.0; // constants for dgemv
	int incx = 1, incy = 1;
	cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, al, A, m, B1, incx, bet, B, incy); //B=A*B1
																					  // declare arrays on the device
	double *d_A, *d_B, *d_tau, *d_work;
	int * devInfo; // device version of info
	int lwork = 0; // workspace size
	int info_gpu = 0; // device info copied to host
	const double one = 1;
	// create cusolver and cublas handles
	cusolver_status = cusolverDnCreate(&cusolverH);
	cublas_status = cublasCreate(&cublasH);
	// prepare memory on the device
	cudaStat1 = cudaMalloc((void **)& d_A, sizeof(double)* lda*m);
	cudaStat2 = cudaMalloc((void **)& d_tau, sizeof(double) * m);
	cudaStat3 = cudaMalloc((void **)& d_B, sizeof(double)* ldb* nrhs);
	cudaStat4 = cudaMalloc((void **)& devInfo, sizeof(int));
	// copy A,B from host to device
	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)* lda*m, cudaMemcpyHostToDevice); // A->d_A
	cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)* ldb*nrhs, cudaMemcpyHostToDevice); // B->d_B

																					  // compute buffer size for geqrf and prepare worksp . on device
	cusolver_status = cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork);
	cudaStat1 = cudaMalloc((void **)& d_work, sizeof(double)* lwork);

	// start timer
	timeStart = steady_clock::now();
	// QR factorization for d_A ; R stored in upper triangle of
	// d_A , elementary reflectors vectors stored in lower triangle
	// of d_A , elementary reflectors scalars stored in d_tau
	cusolver_status = cusolverDnDgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, devInfo);
	cudaStat1 = cudaDeviceSynchronize();

	// stop timer
	timeEnd = steady_clock::now();
	steady_clock::duration cusolverRUN = timeEnd - timeStart;

	//if (debug) {
	//printf(" Dgeqrf time : %lf\n", accum); // print elapsed time
	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); // devInfo -> info_gpu
	
																					 // check error code of geqrf function
	//printf(" after geqrf : info_gpu = %d\n", info_gpu);
	
	// compute d_B =Q^T*B using ormqr function
	cusolver_status = cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda, d_tau, d_B, ldb, d_work, lwork, devInfo);
	cudaStat1 = cudaDeviceSynchronize();

	// devInfo -> info_gpu
	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); 
	
	// check error code of ormqr function
	//printf(" after ormqr : info_gpu = %d\n", info_gpu);
	// write the original system A*X=(Q*R)*X=B in the form
	// R*X=Q^T*B and solve the obtained triangular system
	cublas_status = cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, d_B, ldb);
	cudaStat1 = cudaDeviceSynchronize();
	cudaStat1 = cudaMemcpy(X, d_B, sizeof(double)* ldb*nrhs, cudaMemcpyDeviceToHost); // copy d_B ->X
	//printf(" solution : "); // show first components of the solution
	//for (int i = 0; i < 5; i++) printf("%g, ", X[i]);
	//printf(" ... ");
	//printf("\n");
	//}

	// free memory
	cudaFree(d_A);
	cudaFree(d_tau);
	cudaFree(d_B);
	cudaFree(devInfo);
	cudaFree(d_work);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
	cudaDeviceReset();
	double returnVal = duration_cast<milliseconds>(cusolverRUN).count();
	return returnVal;
}
// Dgeqrf time : 3.333913 sec .
// after geqrf : info_gpu = 0
// after ormqr : info_gpu = 0
// solution : 1, 1, 1, 1, 1, ...