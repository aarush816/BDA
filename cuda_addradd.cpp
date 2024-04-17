
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3

// CUDA kernel to perform matrix addition (A + B)
__global__ void matrix_addition_kernel(int *matrix_a, int *matrix_b, int *result_matrix) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < M) {
        for (int col_idx = 0; col_idx < N; col_idx++) {
            result_matrix[row_idx * N + col_idx] = matrix_a[row_idx * N + col_idx] + matrix_b[row_idx * N + col_idx];
        }
    }
}

// CUDA kernel to perform row-wise sum of a matrix
__global__ void row_addition_kernel(int *matrix, int *row_sums) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < M) {
        int sum = 0;
        for (int col_idx = 0; col_idx < N; col_idx++) {
            sum += matrix[row_idx * N + col_idx];
        }
        row_sums[row_idx] = sum;
    }
}

// CUDA kernel to apply threshold and modify matrix
__global__ void apply_threshold_kernel(int *matrix, int *row_sums, int threshold, int *result_matrix) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < M) {
        if (row_sums[row_idx] < threshold) {
            for (int col_idx = 0; col_idx < N; col_idx++) {
                result_matrix[row_idx * N + col_idx] = 0;
            }
        } else {
            for (int col_idx = 0; col_idx < N; col_idx++) {
                result_matrix[row_idx * N + col_idx] = 1;
            }
        }
    }
}

int main() {
    int matrix_a[M][N];
    int matrix_b[M][N];

    // Initialize random seed
    srand(time(NULL));

    // Generate random matrices A and B
    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i][j] = rand() % 10;  // Generate random value between 0 and 9
            printf("%d ", matrix_a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_b[i][j] = rand() % 10;  // Generate random value between 0 and 9
            printf("%d ", matrix_b[i][j]);
        }
        printf("\n");
    }

    int result_matrix[M][N];

    int *d_matrix_a, *d_matrix_b, *d_result_matrix, *d_sum, *d_modified_matrix;
    int size = M * N * sizeof(int);

    // Allocate memory on device
    cudaMalloc((void **)&d_matrix_a, size);
    cudaMalloc((void **)&d_matrix_b, size);
    cudaMalloc((void **)&d_result_matrix, size);
    cudaMalloc((void **)&d_sum, M * sizeof(int));
    cudaMalloc((void **)&d_modified_matrix, size);

    // Copy matrices from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(4);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x);

    // Perform matrix addition (A + B) using CUDA kernel
    matrix_addition_kernel<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_result_matrix);

    // Copy and print result of matrix addition from device to host
    cudaMemcpy(result_matrix, d_result_matrix, size, cudaMemcpyDeviceToHost);

    printf("\nMatrix A + B (Element-wise Addition):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", result_matrix[i][j]);
        }
        printf("\n");
    }

    // Perform row-wise sum using CUDA kernel
    row_addition_kernel<<<gridSize, blockSize>>>(d_result_matrix, d_sum);

    // Print the threshold value
    int threshold_value = 20;  // Example threshold value
    printf("\nThreshold Value: %d\n", threshold_value);

    // Apply threshold and modify result matrix using CUDA kernel
    apply_threshold_kernel<<<gridSize, blockSize>>>(d_result_matrix, d_sum, threshold_value, d_modified_matrix);

    // Copy modified matrix back from device to host
    cudaMemcpy(result_matrix, d_modified_matrix, size, cudaMemcpyDeviceToHost);

    printf("\nModified Matrix (after thresholding):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", result_matrix[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_result_matrix);
    cudaFree(d_sum);
    cudaFree(d_modified_matrix);

    return 0;
}

