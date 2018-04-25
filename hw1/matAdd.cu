//
// (for GTX970)
// nvcc -arch=compute_52 -code=sm_52,sm_52 -O3 -m64 -o matAdd matAdd.cu
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o matAdd matAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>

// Variables
float* h_A;   // host matrix
float* h_B;
float* h_C;
float* h_D;
float* d_A;   // device matrix
float* d_B;
float* d_C;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void matAdd(float *A, float *B, float *C, int N){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < N && j < N)
        C[i + N*j] = A[i + N*j] + B[i + N*j];
    
    __syncthreads();

}

// Host code

int main( )
{

    int gid;
    int N;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    scanf("%d",&gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("Matrix Addition: C = A + B\n");
    int mem = 1024*1024*1024;     // Giga    

next:
    printf("Enter the size of the matrices: ");       
    scanf("%d",&N);
    printf("%d\n",N);        
    if( 3*N*N > mem ) {     // each real number takes 4 bytes
      printf("The size of these 3 vectors cannot be fitted into 4 Gbyte\n");
      goto next;
    }
    long size = N * N * sizeof(float);


    // Allocate input matrics h_A and h_B in host memory

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Set the sizes of threads and blocks

    int threadsPerBlock;
loop:
    printf("Enter the number of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    if( threadsPerBlock*threadsPerBlock > 1024 ) {
      printf("The number of threads per block must be less than 1024 ! \n");
      goto loop;
    }
    dim3 dimBlock(threadsPerBlock, threadsPerBlock);
    dim3 dimGrid((N + dimBlock.x - 1)/dimBlock.x,
	      	(N + dimBlock.y - 1)/ dimBlock.y);
    printf("The number of blocks is %d\n", dimGrid.x, dimGrid.y);
    if( dimGrid.x > 2147483647 ) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      goto loop;
    }

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    matAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",3*N*N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    h_D = (float*)malloc(size);       // to compute the reference solution
    for (int i = 0; i < N; ++i)
	{
	     for (int j = 0; j < N; ++j)
		h_D[i+ N*j] = h_A[i+ N*j] + h_B[i+ N*j];
	}
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result
    printf("Check result:\n");
    double sum=0; 
    double diff;
    for (int i = 0; i < N; ++i) {
	for(int j = 0; j < N; ++j){
      	   diff = abs(h_D[i+ N*j] - h_C[i+ N*j]);
      	   sum += diff*diff; 
      	   if(diff > 1.0e-15) { 
//           printf("i=%d, j=%d, h_D=%15.10e, h_C=%15.10e \n", i, j, h_D[i+ N*j], h_C[i+ N*j]);
      	   }
	}
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n",sum);

    cudaDeviceReset();
}


// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
	{
	    for (int j = 0; j < n; ++j)
        	data[i+ n*j] = rand() / (float)RAND_MAX;
	}
}



