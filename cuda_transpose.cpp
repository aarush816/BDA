
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 3  // Matrix dimensions


__global__ void matrix_multiplication_kernel(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

__global__ void matrix_addition_kernel(int *a, int *d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        d[row * N + col] = a[row * N + col] + a[col * N + row];
    }
}

__global__ void row_sum_kernel(int *matrix, int *row_sums) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        int sum = 0;
        for (int col = 0; col < N; ++col) {
            sum += matrix[row * N + col];
        }
        row_sums[row] = sum;
    }
}

__global__ void threshold_kernel(int *e, int *f, int threshold, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int sum = e[idx] + f[idx];
        result[idx] = (sum < threshold) ? 0 : 1;
    }
}

__global__ void element_wise_addition_kernel(int *e, int *f, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = e[idx] + f[idx];
    }
}

int main() {
    int a[N][N], b[N][N], c[N][N], d[N][N];
    int e[N], f[N], ef[N], ef_thresholded[N];
    int threshold = 167;  // Example threshold value

    srand(time(NULL));

    printf("Matrix A:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = rand() % 10;  
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i][j] = rand() % 10; 
            printf("%d ", b[i][j]);
        }
        printf("\n");
    }

    int *d_a, *d_b, *d_c, *d_d, *d_e, *d_f, *d_ef, *d_ef_thresholded;
    int size = N * N * sizeof(int);
    int size_rowsum = N * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    cudaMalloc((void **)&d_d, size);
    cudaMalloc((void **)&d_e, size_rowsum);
    cudaMalloc((void **)&d_f, size_rowsum);
    cudaMalloc((void **)&d_ef, size_rowsum);
    cudaMalloc((void **)&d_ef_thresholded, size_rowsum);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c);


    matrix_addition_kernel<<<gridSize, blockSize>>>(d_a, d_d);

    row_sum_kernel<<<(N + blockSize.x - 1) / blockSize.x, blockSize.x>>>(d_c, d_e);
    row_sum_kernel<<<(N + blockSize.x - 1) / blockSize.x, blockSize.x>>>(d_d, d_f);

    element_wise_addition_kernel<<<(N + blockSize.x - 1) / blockSize.x, blockSize.x>>>(d_e, d_f, d_ef);

    threshold_kernel<<<(N + blockSize.x - 1) / blockSize.x, blockSize.x>>>(d_e, d_f, threshold, d_ef_thresholded);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(e, d_e, size_rowsum, cudaMemcpyDeviceToHost);
    cudaMemcpy(f, d_f, size_rowsum, cudaMemcpyDeviceToHost);
    cudaMemcpy(ef, d_ef, size_rowsum, cudaMemcpyDeviceToHost);
    cudaMemcpy(ef_thresholded, d_ef_thresholded, size_rowsum, cudaMemcpyDeviceToHost);


    printf("\nMatrix C (A * B):\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix D (A + A^T):\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", d[i][j]);
        }
        printf("\n");
    }

    printf("\nRow Sums E (of Matrix C):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", e[i]);
    }
    printf("\n");

    printf("\nRow Sums F (of Matrix D):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", f[i]);
    }
    printf("\n");

    printf("\nElement-wise Addition (E + F) Matrix:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", ef[i]);
    }
    printf("\n");

    printf("\nElement-wise Addition (E + F) Thresholded (Threshold = %d):\n", threshold);
    for (int i = 0; i < N; ++i) {
        printf("%d ", ef_thresholded[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
    cudaFree(d_f);
    cudaFree(d_ef);
    cudaFree(d_ef_thresholded);

    return 0;
}

