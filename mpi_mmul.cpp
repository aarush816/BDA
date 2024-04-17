#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 4

void generate_random_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX; // Random value between 0 and 1
    }
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void matrix_multiply(double *matA, double *matB, double *result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += matA[i * size + k] * matB[k * size + j];
            }
            result[i * size + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "This program requires at least 2 MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Determine matrix dimensions
    int rows = MATRIX_SIZE;
    int cols = MATRIX_SIZE;
    int elements_per_process = (rows * cols) / size;

    // Allocate memory for local matrices
    double *local_A = (double *)malloc(elements_per_process * sizeof(double));
    double *local_B = (double *)malloc(cols * elements_per_process * sizeof(double));
    double *local_result = (double *)malloc(elements_per_process * sizeof(double));

    // Generate random matrices on process 0
    if (rank == 0) {
        double *matrix_A = (double *)malloc(rows * cols * sizeof(double));
        double *matrix_B = (double *)malloc(cols * rows * sizeof(double));

        generate_random_matrix(matrix_A, rows, cols);
        generate_random_matrix(matrix_B, cols, rows);

        printf("Matrix A:\n");
        print_matrix(matrix_A, rows, cols);
        printf("\nMatrix B:\n");
        print_matrix(matrix_B, cols, rows);

        // Scatter matrix B to all processes
        MPI_Scatter(matrix_B, elements_per_process * cols, MPI_DOUBLE,
                    local_B, elements_per_process * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        free(matrix_A);
        free(matrix_B);
    } else {
        // Receive scattered matrix B
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL,
                    local_B, elements_per_process * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Broadcast matrix A to all processes
    MPI_Bcast(local_A, elements_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform local matrix multiplication
    matrix_multiply(local_A, local_B, local_result, rows);

    // Gather the results back to process 0
    double *result = NULL;
    if (rank == 0) {
        result = (double *)malloc(rows * cols * sizeof(double));
    }

    MPI_Gather(local_result, elements_per_process, MPI_DOUBLE,
               result, elements_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print the result on process 0
    if (rank == 0) {
        printf("\nMatrix Product (A * B):\n");
        print_matrix(result, rows, cols);
        free(result);
    }

    // Clean up
    free(local_A);
    free(local_B);
    free(local_result);

    MPI_Finalize();
    return 0;
}
