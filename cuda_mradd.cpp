%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3

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

__global__ void apply_threshold(int *matrix_a, int *row_sums, int threshold) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < M) {
        for (int col_idx = 0; col_idx < N; col_idx++) {
            if (row_sums[row_idx] < threshold) {
                matrix_a[row_idx * N + col_idx] = 0; // Set to 0 if row sum is less than threshold
            } else {
                matrix_a[row_idx * N + col_idx] = 1; // Set to 1 if row sum is greater than or equal to threshold
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

    int *d_matrix_a, *d_row_sums;
    int size_a = M * N * sizeof(int);
    int size_row_sums = M * sizeof(int);

    // Allocate device memory for matrix A and row sums
    cudaMalloc((void **)&d_matrix_a, size_a);
    cudaMalloc((void **)&d_row_sums, size_row_sums);

    // Copy matrix A from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size_a, cudaMemcpyHostToDevice);

    // Compute row sums of matrix A on the GPU
    dim3 blockSize_row(256);
    dim3 gridSize_row((M + blockSize_row.x - 1) / blockSize_row.x);

    compute_row_sums<<<gridSize_row, blockSize_row>>>(d_matrix_a, d_row_sums);

    // Threshold value for modification
    int threshold_value = 15;
    printf("\nThreshold Value: %d\n", threshold_value);

    // Apply thresholding based on row sums
    apply_threshold<<<gridSize_row, blockSize_row>>>(d_matrix_a, d_row_sums, threshold_value);

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
    cudaFree(d_row_sums);

    return 0;
}

