#include <stdio.h>
#include <omp.h>

#define N 3  // Matrix dimensions

void matrix_multiplication(int a[][N], int b[][N], int result[][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = 0; // Initialize result element to zero
            for (int k = 0; k < N; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    int a[N][N], b[N][N], result[N][N];

    // Initialize matrices a and b
    printf("Matrix A:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = i * N + j + 1;
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i][j] = (i * N + j + 1) * 10;
            printf("%d ", b[i][j]);
        }
        printf("\n");
    }

    // Perform matrix multiplication using OpenMP
    matrix_multiplication(a, b, result);

    // Print the result matrix
    printf("\nMatrix Product (A * B):\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

    return 0;
}

