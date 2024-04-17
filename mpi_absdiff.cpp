#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4 // Size of the square matrix (N x N)

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Error: Number of processes (%d) must evenly divide matrix size (%d).\n", size, N);
        }
        MPI_Finalize();
        return 1;
    }

    int matrix[N][N];
    int row[N];
    int row_sum;
    int abs_diff[N];

    srand(1234);

    if (rank == 0) {
        // Initialize matrix with random values
        printf("Original Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = rand() % 10;
                printf("%3d ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Scatter rows of the matrix to different processes
    MPI_Scatter(matrix, N * N / size, MPI_INT, row, N, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate row sum and absolute differences
    row_sum = 0;
    for (int i = 0; i < N; i++) {
        row_sum += row[i];
    }

    for (int i = 0; i < N; i++) {
        abs_diff[i] = abs(row_sum - row[i]);
    }

    // Gather absolute differences from all processes to root process
    MPI_Gather(abs_diff, N, MPI_INT, matrix, N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Display the matrix with absolute differences
        printf("Matrix with Absolute Differences:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%3d ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // Display row sums
        printf("Row Sums:\n");
        for (int i = 0; i < N; i++) {
            printf("Row %d: %d\n", i, row_sum);
        }
    }

    MPI_Finalize();
    return 0;
}

