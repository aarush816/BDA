%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3
#define P 3

// CUDA kernel to perform matrix multiplication (A * B)
__global__ void matrix_multiplication_kernel(int *matrix_a, int *matrix_b, int *result_matrix) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < M && col_idx < P) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += matrix_a[row_idx * N + k] * matrix_b[k * P + col_idx];
        }
        result_matrix[row_idx * P + col_idx] = sum;
    }
}

int main() {
    int matrix_a[M][N];
    int matrix_b[N][P];

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
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            matrix_b[i][j] = rand() % 10;
            printf("%d ", matrix_b[i][j]);
        }
        printf("\n");
    }

    int result_multiply[M][P];

    int *d_matrix_a, *d_matrix_b, *d_result_multiply;
    int size_a = M * N * sizeof(int);
    int size_b = N * P * sizeof(int);
    int size_multiply = M * P * sizeof(int);

    // Allocate device memory for matrices and result array
    cudaMalloc((void **)&d_matrix_a, size_a);
    cudaMalloc((void **)&d_matrix_b, size_b);
    cudaMalloc((void **)&d_result_multiply, size_multiply);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, size_b, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for matrix multiplication
    dim3 blockSize(16, 16);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Perform matrix multiplication (A * B) on the GPU
    matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_result_multiply);

    // Copy the result of matrix multiplication from device to host
    cudaMemcpy(result_multiply, d_result_multiply, size_multiply, cudaMemcpyDeviceToHost);

    // Display the result of matrix multiplication
    printf("\nMatrix A * B (Result of Matrix Multiplication):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d ", result_multiply[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_result_multiply);

    return 0;
}
