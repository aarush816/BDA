%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3

// CUDA kernel to perform matrix addition (A + B)
__global__ void matrix_addition_kernel(int *matrix_a, int *matrix_b, int *result_matrix) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < M && col_idx < N) {
        result_matrix[row_idx * N + col_idx] = matrix_a[row_idx * N + col_idx] + matrix_b[row_idx * N + col_idx];
    }
}

int main() {
    int matrix_a[M][N];
    int matrix_b[M][N];

    srand(time(NULL));

    // Initialize matrices A and B with random values
    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i][j] = rand() % 10;
            printf("%d ", matrix_a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_b[i][j] = rand() % 10;
            printf("%d ", matrix_b[i][j]);
        }
        printf("\n");
    }

    int result_add[M][N];

    int *d_matrix_a, *d_matrix_b, *d_result_add;
    int size = M * N * sizeof(int);

    // Allocate device memory for matrices and result array
    cudaMalloc((void **)&d_matrix_a, size);
    cudaMalloc((void **)&d_matrix_b, size);
    cudaMalloc((void **)&d_result_add, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for matrix addition
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Perform matrix addition (A + B) on the GPU
    matrix_addition_kernel<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_result_add);

    // Copy the result of matrix addition from device to host
    cudaMemcpy(result_add, d_result_add, size, cudaMemcpyDeviceToHost);

    // Display the result of matrix addition
    printf("\nMatrix A + B (Result of Matrix Addition):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", result_add[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_result_add);

    return 0;
}

