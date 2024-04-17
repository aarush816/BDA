%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3

// CUDA kernel to compute column sums of matrix A
__global__ void compute_column_sums(int *matrix_a, int *col_sums) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx < N) {
        int sum = 0;
        for (int row_idx = 0; row_idx < M; row_idx++) {
            sum += matrix_a[row_idx * N + col_idx];
        }
        col_sums[col_idx] = sum;
    }
}

// CUDA kernel to apply threshold and modify result array based on column sums
__global__ void apply_threshold_kernel(int *col_sums, int threshold, int *result_array) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx < N) {
        if (col_sums[col_idx] < threshold) {
            result_array[col_idx] = 0;
        } else {
            result_array[col_idx] = 1;
        }
    }
}

int main() {
    int matrix_a[M][N];

    // Initialize random seed
    srand(time(NULL));

    // Generate random matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i][j] = rand() % 10;  // Generate random value between 0 and 9
            printf("%d ", matrix_a[i][j]);
        }
        printf("\n");
    }

    int col_sums[N];
    int result_array[N];

    int *d_matrix_a, *d_col_sums, *d_result_array;
    int size_matrix_a = M * N * sizeof(int);
    int size_col_sums = N * sizeof(int);
    int size_result_array = N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void **)&d_matrix_a, size_matrix_a);
    cudaMalloc((void **)&d_col_sums, size_col_sums);
    cudaMalloc((void **)&d_result_array, size_result_array);

    // Copy matrix A from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size_matrix_a, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for column sum computation
    dim3 blockSize_col(256);
    dim3 gridSize_col((N + blockSize_col.x - 1) / blockSize_col.x);

    // Compute column sums of matrix A on the GPU
    compute_column_sums<<<gridSize_col, blockSize_col>>>(d_matrix_a, d_col_sums);

    // Define threshold value
    int threshold_value = 20;  // Example threshold value
    printf("\nThreshold Value: %d\n", threshold_value);

    // Apply threshold and modify result array on the GPU
    apply_threshold_kernel<<<gridSize_col, blockSize_col>>>(d_col_sums, threshold_value, d_result_array);

    // Copy result array from device to host
    cudaMemcpy(result_array, d_result_array, size_result_array, cudaMemcpyDeviceToHost);

    // Display the modified result array after thresholding
    printf("\nModified Result Array (after column sum thresholding):\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", result_array[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_matrix_a);
    cudaFree(d_col_sums);
    cudaFree(d_result_array);

    return 0;
}

