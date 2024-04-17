%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 3
#define N 3
#define P 3

// CUDA kernel to perform matrix addition (A + B)
__global__ void matrix_addition_kernel(int *matrix_a, int *matrix_b, int *result_matrix) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < M && col_idx < N) {
        result_matrix[row_idx * N + col_idx] = matrix_a[row_idx * N + col_idx] + matrix_b[row_idx * N + col_idx];
    }
}

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

    int result_add[M][N];
    int result_multiply[M][P];

    int *d_matrix_a, *d_matrix_b, *d_result_add, *d_result_multiply;

    int size_a = M * N * sizeof(int);
    int size_b = N * P * sizeof(int);
    int size_add = M * N * sizeof(int);
    int size_multiply = M * P * sizeof(int);

    // Allocate device memory for matrices and result arrays
    cudaMalloc((void **)&d_matrix_a, size_a);
    cudaMalloc((void **)&d_matrix_b, size_b);
    cudaMalloc((void **)&d_result_add, size_add);
    cudaMalloc((void **)&d_result_multiply, size_multiply);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_matrix_a, matrix_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, size_b, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for matrix addition
    dim3 blockSize_add(16, 16);
    dim3 gridSize_add((N + blockSize_add.x - 1) / blockSize_add.x, (M + blockSize_add.y - 1) / blockSize_add.y);

    // Perform matrix addition (A + B) on the GPU
    matrix_addition_kernel<<<gridSize_add, blockSize_add>>>(d_matrix_a, d_matrix_b, d_result_add);

    // Copy the result of matrix addition from device to host
    cudaMemcpy(result_add, d_result_add, size_add, cudaMemcpyDeviceToHost);

    // Display the result of matrix addition
    printf("\nMatrix A + B (Result of Matrix Addition):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", result_add[i][j]);
        }
        printf("\n");
    }

    // Define grid and block dimensions for matrix multiplication
    dim3 blockSize_multiply(16, 16);
    dim3 gridSize_multiply((P + blockSize_multiply.x - 1) / blockSize_multiply.x, (M + blockSize_multiply.y - 1) / blockSize_multiply.y);

    // Perform matrix multiplication (A * B) on the GPU
    matrix_multiplication_kernel<<<gridSize_multiply, blockSize_multiply>>>(d_matrix_a, d_matrix_b, d_result_multiply);

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
    cudaFree(d_result_add);
    cudaFree(d_result_multiply);

    return 0;
}

