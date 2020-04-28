//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 1
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="Taiwan_No1";
	std::string author_1="Hsu_Cheng";
	std::string author_2="Andrew_Hederman";
	std::string author_3="Name_3";	////optional
};

////This is a matrix class to carry out linear algebra operations on both GPU and CPU
////It is the same as the sample code I showed in class on Week 3. 

////NOTICE: You do not have to change the implementation in this class. 
////But if you do want to change part of it for performance reasons, please let us known by writting a submission note on Canvas.

class Matrix{
public:
    int m=0;							////number of rows
    int n=0;							////number of columns
	vector<float> elements_on_host;		////we use a std::vector for the element array on host
    float* elements_on_dev=0;			////we use a pointer for the element array on device
	bool on_host=true;

	////constructors
	__host__ Matrix(){}

	__host__ Matrix(const int _m,const int _n,bool _on_host=true)
	{
		on_host=_on_host;
		if(on_host)Resize_On_Host(_m,_n);
		else Resize_On_Device(_m,_n);
	}

	////destructor
	__host__ ~Matrix()
	{
		if(!on_host&&elements_on_dev!=0) cudaFree(elements_on_dev);		
	}

	////Resize on host or device
	__host__ void Resize_On_Host(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		elements_on_host.resize(m*n);
	}

	__host__ void Resize_On_Device(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		if(elements_on_dev!=0)cudaFree(elements_on_dev);
		cudaMalloc((void**)&elements_on_dev,m*n*sizeof(float));
	}

	////random access a matrix element
	inline __host__ float& operator() (const int i,const int j)
	{
		return elements_on_host[i*n+j];
	}

	inline __host__ const float& operator() (const int i,const int j) const
	{
		return elements_on_host[i*n+j];
	}

	////copy data with four cases (CPU->CPU, GPU->CPU, GPU->GPU, CPU->GPU)
	__host__ Matrix& operator= (const Matrix& mtx)
	{
		if(on_host&&mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			elements_on_host=mtx.elements_on_host;
		}
		else if(on_host&&!mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			cudaMemcpy(&elements_on_host[0],mtx.elements_on_dev,m*n*sizeof(float),cudaMemcpyDeviceToHost);
		}
		else if(!on_host&&!mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,mtx.elements_on_dev,mtx.m*n*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		else if(!on_host&&mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,&mtx.elements_on_host[0],m*n*sizeof(float),cudaMemcpyHostToDevice);
		}
		return *this;
	}

	////print matrix elements on screen
	__host__ friend ostream & operator << (ostream &out,const Matrix &mtx)
	{
		if(!mtx.on_host)
			cout<<"Print for matrix on device is not supported."<<endl;

		for(int i=0;i<mtx.m;i++){
			for(int j=0;j<mtx.n;j++){
				out<<mtx(i,j)<<", ";
			}
			out<<std::endl;
		}
		return out;
	}
};

//////////////////////////////////////////////////////////////////////////
////Your tasks start!

////This is a sample implementation without using any memory hierarchy
////The function calculates C=A*B, with dimA=[Am,An], dimB=[Bm,Bn], dimC=[Am,bn], and An=Bm
__global__ void Matrix_Multiplication_AB_Kernel_Poorman(const float* Ae,const float* Be,float* Ce,const int Am,const int An,const int Bn)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;

	float val=0.f;
	for(int k=0;k<An;k++)
		val+=Ae[i*An+k]*Be[k*Bn+j];
	Ce[i*Bn+j]=val;
	
} 

//////////////////////////////////////////////////////////////////////////
////Task 1: implement your fast matrix-matrix multiplication in the following kernel function.
////The function parameters are the same as the sample function
//////////////////////////////////////////////////////////////////////////

#define TILE_SIZE 16

/*Your may want to declare your global variables here*/
__global__ void Matrix_Multiplication_AB_Kernel_Your_Version(const float* Ae,const float* Be,float* Ce,const int Am,const int An,const int Bn)
{
	/*Your implementation starts*/
	
	// Block index

	__shared__ float shared_a[TILE_SIZE][TILE_SIZE];
	__shared__ float shared_b[TILE_SIZE][TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int a_rows = Am;
	int a_columns = An;
	int b_columns = Bn;

	//check if thread directly maps to the dimensions of the resulting matrix
	if (row < a_rows && col < b_columns){
		float result = 0;
		
		for (int phase = 0; phase <= a_columns / TILE_SIZE; phase++){
			shared_a[ty][tx] = Ae[row * a_columns + phase * TILE_SIZE + tx];
			shared_b[ty][tx] = Be[(phase * TILE_SIZE + ty) * b_columns + col];
			__syncthreads();

			for (int k = 0; k < TILE_SIZE; k++){
				if (k + (phase * TILE_SIZE) < a_columns){
					result += (shared_a[ty][k] * shared_b[k][tx]);
				}
			}
			__syncthreads();
		}
		
		
		Ce[row * b_columns + col] = result;
	}
	
	/*Your implementation ends*/
}

////This is a sample implementation without using any memory hierarchy
////The function calculates the matrix multiplication, with C=A^T*B*A, A^T is the transpose of A, dimA=[Am,An], dimB=[Am,Am], and dimC=[An,An]
__global__ void Matrix_Multiplication_ATBA_Kernel_Poorman(const float* Ae,const float* Be,float* Ce,const int Am,const int An)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	
	float val=0.f;
	for(int l=0;l<Am;l++)
		for(int k=k=0;k<Am;k++)
			val+=Ae[l*An+i]*Be[l*Am+k]*Ae[k*An+j];
	Ce[i*An+j]=val;
}


// ref: https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
__global__ void transpose(float* AT, float* A, int width, int height)
{
	//A(m, n), width = m, height = n
	__shared__ float s_block[TILE_SIZE][TILE_SIZE];
	
	
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;

	if ((x < width) && (y < height)){
		int i = y * width + x;
		s_block[threadIdx.y][threadIdx.x] = A[i];
	}

	__syncthreads();

	x = blockIdx.y * TILE_SIZE + threadIdx.x;
	y = blockIdx.x * TILE_SIZE + threadIdx.y;

	if ((x < height) && (y < width)){
		int i = y * height + x;
		AT[i] = s_block[threadIdx.x][threadIdx.y];
	}
}

//////////////////////////////////////////////////////////////////////////
////Task 2: calculate the matrix multiplication in the following kernel function. 
////The function parameters are the same as the sample function
//////////////////////////////////////////////////////////////////////////

__global__ void Matrix_Multiplication_ATBA_Kernel_Your_Version(const float* Ae,const float* Be,float* Ce,const int Am,const int An)
{
	/*Your implementation starts*/
	
	/*Your implementation ends*/
}

//////////////////////////////////////////////////////////////////////////
////Task 3:  calculate the Frobenius norm of a matrix
////The definition of F-norm for a matrix is square root of (the sum of squares of all the matrix elements), i.e., F=sqrt(sum_(A_ij^2))
////See the definition: https://mathworld.wolfram.com/FrobeniusNorm.html
//////////////////////////////////////////////////////////////////////////

////Please write your own kernel function here, and call it in the function Test_F_Norm_On_GPU to test its correctness and performance
/*Your implementation starts*/

__global__ void Get_F_Norm(const float* A, float* sum)
{
	
	__shared__ float sdata[256];
	//printf("sdata =\n");
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	

	sdata[threadIdx.x] = A[i]*A[i];
	

	__syncthreads();
	// do reduction in shared mem
	for (int j = 1; j < blockDim.x; j *= 2){
		int index = 2 * j * threadIdx.x;

		if (index < blockDim.x){
			sdata[index] += sdata[index + j];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (threadIdx.x == 0)
		atomicAdd(sum, (sdata[0]));
	
	
	//printf("val = %f\n", sum[0]);
	
	/*
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int i = bid * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

	float val = 0.f;
	for (int k = 0; k < i; k++)
		sum[0] += (A[i] * A[i]);
	
	sum[0] = sqrt(sum[0]);
	printf("val = %f\n", A[i]);
	*/
	
}
/*Your implementation ends*/





////Congratulations, your tasks are all finished!
//////////////////////////////////////////////////////////////////////////


////Here are the test functions for your three kernel implementations

ofstream out;

__host__ void Test_Matrix_Multiplication_AB_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;

	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.m,B_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size=16;
	const int block_num_x=C.m/block_size;
	const int block_num_y=C.n/block_size;

	////TODO: this is a sample implementation. Comment it out to test your own code.
	//Matrix_Multiplication_AB_Kernel_Poorman<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
	//	(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);

	////TODO: Uncomment this to test your own implementation
	////NOTICE: You do not have to use the block_size I specified here. You may customize the size of your grid and blocks for better performance.

	Matrix_Multiplication_AB_Kernel_Your_Version<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication AB: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);	
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	
	C=C_on_dev;
	
	
	out<<"T1: "<<gpu_time<<endl;
}

__host__ void Test_Matrix_Multiplication_ATBA_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;

	Matrix AT_on_dev(A.m, A.n, false);
	AT_on_dev = A;
	
	//// Allocate C in device memory
	Matrix tmp_on_dev(A_on_dev.m, A_on_dev.n, false);

	Matrix C_on_dev(A_on_dev.n,A_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size=16;
	const int block_num_x=C.m/block_size;
	const int block_num_y=C.n/block_size;

	////TODO: this is a sample implementation. Comment it out to test your own code.
	//Matrix_Multiplication_ATBA_Kernel_Poorman<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		//(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);

	////TODO: Uncomment this to test your own implementation.
	////NOTICE: You do not have to use the block_size I specified here. You may customize the size of your grid and blocks for better performance.
	transpose << <dim3(block_num_x, block_num_y), dim3(block_size, block_size) >> >
		(AT_on_dev.elements_on_dev, A_on_dev.elements_on_dev, A_on_dev.m, A_on_dev.n);

	Matrix_Multiplication_AB_Kernel_Your_Version << <dim3(block_num_x, block_num_y), dim3(block_size, block_size) >> >
		(B_on_dev.elements_on_dev, A_on_dev.elements_on_dev, tmp_on_dev.elements_on_dev, B_on_dev.m, B_on_dev.n, A_on_dev.n);

	Matrix_Multiplication_AB_Kernel_Your_Version << <dim3(block_num_x, block_num_y), dim3(block_size, block_size) >> >
		(AT_on_dev.elements_on_dev, tmp_on_dev.elements_on_dev, C_on_dev.elements_on_dev, A_on_dev.n, A_on_dev.m, A_on_dev.n);
	//Matrix_Multiplication_ATBA_Kernel_Your_Version<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
	//	(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication ATBA: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	C=C_on_dev;

	

	out<<"T2: "<<gpu_time<<endl;
}

__host__ void Test_Matrix_F_Norm_On_GPU(const Matrix& A, float& norm)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;


	const float a_host[1] = {0};

	float* sum_dev = nullptr;
	cudaMalloc((void**)&sum_dev, 1 * sizeof(float));
	cudaMemcpy(sum_dev, a_host, 1 * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size = 16;
	const int block_num_x = A.m / block_size;
	const int block_num_y = A.n / block_size;

	////TODO: call the F norm kernel you implemented, and sum_dev the value to the passed-in variable norm
	Get_F_Norm<<<dim3(block_num_x * block_num_y), dim3(block_size*block_size) >>>(A_on_dev.elements_on_dev, sum_dev);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "f-norm failed: %s\n", cudaGetErrorString(cudaStatus));
		
	}
	
	cudaMemcpy(&norm, &sum_dev[0], 1 * sizeof(float), cudaMemcpyDeviceToHost);

	norm = sqrt(norm);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for F norm: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	out<<"T3: "<<gpu_time<<endl;
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_1_matrix.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	////NOTICE: We may use a different set of parameters to evaluate your code.
	////So please test your functions with different size and initial values.
	//////////////////////////////////////////////////////////////////////////

	const int m = 512;
	const int n = 2048;
	const int p = 1024;

	Matrix h_A(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			h_A(i, j) = 1.f;
		}
	}

	Matrix h_B(n, p);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			h_B(i, j) = 1.f;
		}
	}

	Matrix h_C(m, p);

	Matrix h_B2(m, m);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			h_B2(i, j) = 1.f;
		}
	}

	Matrix h_C2(n, n);

	Test_Matrix_Multiplication_AB_On_GPU(h_A, h_B, h_C);
	cout << "AB result: " << h_C(h_C.m / 2, h_C.n / 2) << endl;
	out << "R1: " << h_C(h_C.m / 2, h_C.n / 2) << endl;

	Test_Matrix_Multiplication_ATBA_On_GPU(h_A, h_B2, h_C2);
	cout << "ATBA result: " << h_C2(h_C2.m / 3, h_C2.n / 3) << endl;
	out << "R2: " << h_C2(h_C2.m / 3, h_C2.n / 3) << endl;

	float f_norm = 0.f;
	Test_Matrix_F_Norm_On_GPU(h_A, f_norm);
	cout << "F-norm result: " << f_norm << endl;
	out << "R3: " << f_norm << endl;

	return 0;
}