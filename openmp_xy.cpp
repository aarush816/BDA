#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MATRIX_SIZE 4

int main() {
    int i, j;
    int x = 3;
    int y = 2;

    int A[MATRIX_SIZE][MATRIX_SIZE];
    int B[MATRIX_SIZE][MATRIX_SIZE];
    int C[MATRIX_SIZE][MATRIX_SIZE];
    int updated_C[MATRIX_SIZE][MATRIX_SIZE];

    srand(1234);

    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            A[i][j] = rand() % 10;
        }
    }

    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            B[i][j] = rand() % 10;
        }
    }

    #pragma omp parallel for private(i, j) shared(A, B, C, x, y, updated_C)
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            C[i][j] = A[i][j] + B[i][j];

            if (C[i][j] % 2 == 0) {
                updated_C[i][j] = C[i][j] - x;
            } else {
                updated_C[i][j] = C[i][j] + y;
            }
        }
    }

    printf("Matrix C (Result):\n");
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            printf("%4d ", C[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Updated Matrix C (After Conditions):\n");
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            printf("%4d ", updated_C[i][j]);
        }
        printf("\n");
    }

    return 0;
}

