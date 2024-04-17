
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 4
#define N 3
#define P 3


__global__ void matrix_addition_kernel(int *matrix_a, int *matrix_b, int *result_matrix) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < M && col_idx < N) {
        result_matrix[row_idx * N + col_idx] = matrix_a[row_idx * N + col_idx] + matrix_b[row_idx * N + col_idx];
    }
}


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

__global__ void row_sum_kernel(int *matrix, int *row_sums) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < M) {
        int sum = 0;
        for (int col_idx = 0; col_idx < N; col_idx++) {
            sum += matrix[row_idx * N + col_idx];
        }
        row_sums[row_idx] = sum;
    }
}


__global__ void column_sum_kernel(int *matrix, int *col_sums) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx < N) {
        int sum = 0;
        for (int row_idx = 0; row_idx < M; row_idx++) {
            sum += matrix[row_idx * N + col_idx];
        }
        col_sums[col_idx] = sum;
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

    int result_add[M][N];
    int result_multiply[M][P];
    int row_sums_add[M];
    int row_sums_multiply[M];
    int col_sums_add[N];
    int col_sums_multiply[P];

    int *d_matrix_a, *d_matrix_b, *d_result_add, *d_result_multiply;
    int *d_row_sums_add, *d_row_sums_multiply;
    int *d_col_sums_add, *d_col_sums_multiply;

    int size_a = M * N * sizeof(int);
    int size_b = N * P * sizeof(int);
    int size_add = M * N * sizeof(int);
    int size_multiply = M * P * sizeof(int);
    int size_row_sums = M * sizeof(int);
    int size_col_sums_add = N * sizeof(int);
    int size_col_sums_multiply = P * sizeof(int);


    cudaMalloc((void **)&d_matrix_a, size_a);
    cudaMalloc((void **)&d_matrix_b, size_b);
    cudaMalloc((void **)&d_result_add, size_add);
    cudaMalloc((void **)&d_result_multiply, size_multiply);
    cudaMalloc((void **)&d_row_sums_add, size_row_sums);
    cudaMalloc((void **)&d_row_sums_multiply, size_row_sums);
    cudaMalloc((void **)&d_col_sums_add, size_col_sums_add);
    cudaMalloc((void **)&d_col_sums_multiply, size_col_sums_multiply);


    cudaMemcpy(d_matrix_a, matrix_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, size_b, cudaMemcpyHostToDevice);


    dim3 blockSize_add(16, 16);
    dim3 gridSize_add((N + blockSize_add.x - 1) / blockSize_add.x, (M + blockSize_add.y - 1) / blockSize_add.y);

   
    matrix_addition_kernel<<<gridSize_add, blockSize_add>>>(d_matrix_a, d_matrix_b, d_result_add);


    cudaMemcpy(result_add, d_result_add, size_add, cudaMemcpyDeviceToHost);


    row_sum_kernel<<<(M + blockSize_add.x - 1) / blockSize_add.x, blockSize_add.x>>>(d_result_add, d_row_sums_add);

  
    column_sum_kernel<<<(N + blockSize_add.x - 1) / blockSize_add.x, blockSize_add.x>>>(d_result_add, d_col_sums_add);

   
    printf("\nMatrix A + B (Result of Matrix Addition):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", result_add[i][j]);
        }
        printf("\n");
    }

    cudaMemcpy(row_sums_add, d_row_sums_add, size_row_sums, cudaMemcpyDeviceToHost);
    printf("\nRow Sums of A + B:\n");
    for (int i = 0; i < M; i++) {
        printf("%d ", row_sums_add[i]);
    }
    printf("\n");

    cudaMemcpy(col_sums_add, d_col_sums_add, size_col_sums_add, cudaMemcpyDeviceToHost);
    printf("\nColumn Sums of A + B:\n");
    for (int j = 0; j < N; j++) {
        printf("%d ", col_sums_add[j]);
    }
    printf("\n");

    
    dim3 blockSize_multiply(16, 16);
    dim3 gridSize_multiply((P + blockSize_multiply.x - 1) / blockSize_multiply.x, (M + blockSize_multiply.y - 1) / blockSize_multiply.y);

   
    matrix_multiplication_kernel<<<gridSize_multiply, blockSize_multiply>>>(d_matrix_a, d_matrix_b, d_result_multiply);

    
    cudaMemcpy(result_multiply, d_result_multiply, size_multiply, cudaMemcpyDeviceToHost);

   
    row_sum_kernel<<<(M + blockSize_multiply.x - 1) / blockSize_multiply.x, blockSize_multiply.x>>>(d_result_multiply, d_row_sums_multiply);

    
    column_sum_kernel<<<(P + blockSize_multiply.x - 1) / blockSize_multiply.x, blockSize_multiply.x>>>(d_result_multiply, d_col_sums_multiply);

   
    printf("\nMatrix A * B (Result of Matrix Multiplication):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d ", result_multiply[i][j]);
        }
        printf("\n");
    }

    cudaMemcpy(row_sums_multiply, d_row_sums_multiply, size_row_sums, cudaMemcpyDeviceToHost);
    printf("\nRow Sums of A * B:\n");
    for (int i = 0; i < M; i++) {
        printf("%d ", row_sums_multiply[i]);
    }
    printf("\n");

    cudaMemcpy(col_sums_multiply, d_col_sums_multiply, size_col_sums_multiply, cudaMemcpyDeviceToHost);
    printf("\nColumn Sums of A * B:\n");
    for (int j = 0; j < P; j++) {
        printf("%d ", col_sums_multiply[j]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_result_add);
    cudaFree(d_result_multiply);
    cudaFree(d_row_sums_add);
    cudaFree(d_row_sums_multiply);
    cudaFree(d_col_sums_add);
    cudaFree(d_col_sums_multiply);

    return 0;
}

