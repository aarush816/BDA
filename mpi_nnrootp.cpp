#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4  // Size of the matrix (N x N)

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure number of processes matches the size of the matrix
    if (size != N) {
        if (rank == 0) {
            printf("Error: Number of processes must match the size of the matrix (N x N = %d x %d)\n", N, N);
        }
        MPI_Finalize();
        return 1;
    }

    // Root process (rank 0) generates and distributes the matrix
    int matrix[N][N];
    if (rank == 0) {
        printf("Original Matrix:\n");
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = rand() % 10; // Fill matrix with random numbers (0-9)
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }

    // Scatter the matrix rows across all processes
    int row[N];
    MPI_Scatter(matrix, N, MPI_INT, row, N, MPI_INT, 0, MPI_COMM_WORLD);

    // Process each row independently
    for (int i = 0; i < N; i++) {
        row[i] = (row[i] % 2 == 0) ? 0 : 1; // Apply even/odd transformation
    }

    // Gather the modified rows back to the root process
    MPI_Gather(row, N, MPI_INT, matrix, N, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the modified matrix from the root process
    if (rank == 0) {
        printf("\nModified Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}

