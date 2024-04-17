%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3

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

__global__ void apply_threshold(int *matrix_a, int *col_sums, int threshold) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx < N) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            if (col_sums[col_idx] < threshold) {
                matrix_a[row_idx * N + col_idx] = 0; // Set to 0 if column sum is less than threshold
            } else {
                matrix_a[row_idx * N + col_idx] = 1; // Set to 1 if column sum is greater than or equal to threshold
            }
        }
    }
}

int main() {
    int matrix_a[M][N];

    srand(time(NULL));

    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i][j] = rand() % 10;
            printf("%d ", matrix_a[i][j]);
        }
        printf("\n");
    }

    int *d_matrix_a, *d_col_sums;
    int size_a = M * N * sizeof(int);
    int size_col_sums = N * sizeof(int);

    // Allocate device memory for matrix A and column sums
    cudaMalloc((void **)&d_matrix_a, size_a);
    cudaMalloc((void **)&d_col_sums, size_col_sums);

    // Copy matrix A from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size_a, cudaMemcpyHostToDevice);

    // Compute column sums of matrix A on the GPU
    dim3 blockSize_col(256);
    dim3 gridSize_col((N + blockSize_col.x - 1) / blockSize_col.x);

    compute_column_sums<<<gridSize_col, blockSize_col>>>(d_matrix_a, d_col_sums);

    // Threshold value for modification
    int threshold_value = 15;
    printf("\nThreshold Value: %d\n", threshold_value);

    // Apply thresholding based on column sums
    apply_threshold<<<gridSize_col, blockSize_col>>>(d_matrix_a, d_col_sums, threshold_value);

    // Copy modified matrix A back from device to host
    cudaMemcpy(matrix_a, d_matrix_a, size_a, cudaMemcpyDeviceToHost);

    // Display the modified matrix A after thresholding
    printf("\nModified Matrix A (after thresholding):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix_a[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_matrix_a);
    cudaFree(d_col_sums);

    return 0;
}

