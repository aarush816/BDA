#include <stdio.h>
#include <mpi.h>

#define MAX_SIZE 10  // Maximum size for the matrix (adjust as needed)

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    int matrix[MAX_SIZE][MAX_SIZE];
    int local_matrix[MAX_SIZE][MAX_SIZE];

    if (rank == 0) {
        // Root process prompts user to enter matrix size
        printf("Enter the size of the matrix (n x n, max size %d): ", MAX_SIZE);
        scanf("%d", &n);

        // Validate matrix size
        if (n <= 0 || n > MAX_SIZE) {
            printf("Invalid matrix size. Please enter a size between 1 and %d.\n", MAX_SIZE);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast matrix size from root process to all other processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate number of rows each process will handle
    int rows_per_process = n / size;
    int remainder = n % size;

    // Scatter matrix rows from root process to all processes
    MPI_Scatter(matrix, rows_per_process * n, MPI_INT,
                local_matrix, rows_per_process * n, MPI_INT,
                0, MPI_COMM_WORLD);

    // Modify local matrix elements based on even/odd condition
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < n; j++) {
            if (local_matrix[i][j] % 2 == 0) {
                local_matrix[i][j] = 0;  // Replace even element with 0
            } else {
                local_matrix[i][j] = 1;  // Replace odd element with 1
            }
        }
    }

    // Gather modified matrix rows back to root process
    MPI_Gather(local_matrix, rows_per_process * n, MPI_INT,
               matrix, rows_per_process * n, MPI_INT,
               0, MPI_COMM_WORLD);

    // Root process displays the modified matrix
    if (rank == 0) {
        printf("Matrix with even elements replaced by 0 and odd elements replaced by 1:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}

