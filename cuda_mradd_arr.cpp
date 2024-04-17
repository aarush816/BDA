%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3

// CUDA kernel to compute row sums of matrix A
__global__ void compute_row_sums(int *matrix_a, int *row_sums) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < M) {
        int sum = 0;
        for (int col_idx = 0; col_idx < N; col_idx++) {
            sum += matrix_a[row_idx * N + col_idx];
        }
        row_sums[row_idx] = sum;
    }
}

// CUDA kernel to apply threshold and modify result array based on row sums
__global__ void apply_threshold_kernel(int *row_sums, int threshold, int *result_array) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < M) {
        if (row_sums[row_idx] < threshold) {
            result_array[row_idx] = 0;
        } else {
            result_array[row_idx] = 1;
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

    int row_sums[M];
    int result_array[M];

    int *d_matrix_a, *d_row_sums, *d_result_array;
    int size_matrix_a = M * N * sizeof(int);
    int size_row_sums = M * sizeof(int);
    int size_result_array = M * sizeof(int);

    // Allocate device memory
    cudaMalloc((void **)&d_matrix_a, size_matrix_a);
    cudaMalloc((void **)&d_row_sums, size_row_sums);
    cudaMalloc((void **)&d_result_array, size_result_array);

    // Copy matrix A from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size_matrix_a, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for row sum computation
    dim3 blockSize_row(256);
    dim3 gridSize_row((M + blockSize_row.x - 1) / blockSize_row.x);

    // Compute row sums of matrix A on the GPU
    compute_row_sums<<<gridSize_row, blockSize_row>>>(d_matrix_a, d_row_sums);

    // Define threshold value
    int threshold_value = 15;  // Example threshold value
    printf("\nThreshold Value: %d\n", threshold_value);

    // Apply threshold and modify result array on the GPU
    apply_threshold_kernel<<<gridSize_row, blockSize_row>>>(d_row_sums, threshold_value, d_result_array);

    // Copy result array from device to host
    cudaMemcpy(result_array, d_result_array, size_result_array, cudaMemcpyDeviceToHost);

    // Display the modified result array after thresholding
    printf("\nModified Result Array (after row sum thresholding):\n");
    for (int i = 0; i < M; i++) {
        printf("%d ", result_array[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_matrix_a);
    cudaFree(d_row_sums);
    cudaFree(d_result_array);

    return 0;
}

