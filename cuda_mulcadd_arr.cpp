
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 3
#define N 3
#define P 3

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

__global__ void column_addition_kernel(int *matrix, int *col_sums) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx < P) {
        int sum = 0;
        for (int row_idx = 0; row_idx < M; row_idx++) {
            sum += matrix[row_idx * P + col_idx];
        }
        col_sums[col_idx] = sum;
    }
}

__global__ void apply_threshold_column_kernel(int *matrix, int *col_sums, int threshold, int *result_matrix) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx < P) {
        // Determine threshold result for the current column
        result_matrix[col_idx] = (col_sums[col_idx] < threshold) ? 0 : 1;
    }
}

int main() {
    int matrix_a[M][N];
    int matrix_b[N][P];

    srand(time(NULL));

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

    int result_matrix[M][P];

    int *d_matrix_a, *d_matrix_b, *d_result_matrix, *d_col_sums, *d_thresholded_matrix;
    int size_a = M * N * sizeof(int);
    int size_b = N * P * sizeof(int);
    int size_result = M * P * sizeof(int);
    int size_col_sums = P * sizeof(int);
    int size_thresholded = P * sizeof(int);

    cudaMalloc((void **)&d_matrix_a, size_a);
    cudaMalloc((void **)&d_matrix_b, size_b);
    cudaMalloc((void **)&d_result_matrix, size_result);
    cudaMalloc((void **)&d_col_sums, size_col_sums);
    cudaMalloc((void **)&d_thresholded_matrix, size_thresholded);

    cudaMemcpy(d_matrix_a, matrix_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, size_b, cudaMemcpyHostToDevice);

    dim3 blockSize_mul(16, 16);
    dim3 gridSize_mul((P + blockSize_mul.x - 1) / blockSize_mul.x, (M + blockSize_mul.y - 1) / blockSize_mul.y);

    matrix_multiplication_kernel<<<gridSize_mul, blockSize_mul>>>(d_matrix_a, d_matrix_b, d_result_matrix);

    cudaMemcpy(result_matrix, d_result_matrix, size_result, cudaMemcpyDeviceToHost);

    printf("\nMatrix A * B (Result of Matrix Multiplication):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d ", result_matrix[i][j]);
        }
        printf("\n");
    }

    dim3 blockSize_col(256);
    dim3 gridSize_col((P + blockSize_col.x - 1) / blockSize_col.x);

    column_addition_kernel<<<gridSize_col, blockSize_col>>>(d_result_matrix, d_col_sums);

    int threshold_value = 150;
    printf("\nThreshold Value: %d\n", threshold_value);

    apply_threshold_column_kernel<<<gridSize_col, blockSize_col>>>(d_result_matrix, d_col_sums, threshold_value, d_thresholded_matrix);

    int thresholded_result[P];
    cudaMemcpy(thresholded_result, d_thresholded_matrix, size_thresholded, cudaMemcpyDeviceToHost);

    printf("\nThresholded Matrix (after column sum thresholding):\n");
    for (int j = 0; j < P; j++) {
        printf("%d ", thresholded_result[j]);
    }
    printf("\n");

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_result_matrix);
    cudaFree(d_col_sums);
    cudaFree(d_thresholded_matrix);

    return 0;
}

