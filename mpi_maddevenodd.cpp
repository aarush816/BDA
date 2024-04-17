#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4 // Size of the square matrices (N x N)

void add_matrices(int local_A[N][N], int local_B[N][N], int local_C[N][N], int size) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = local_A[i][j] + local_B[i][j];
        }
    }
}

void apply_element_transformation(int local_C[N][N], int size) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (local_C[i][j] % 2 == 0) {
                local_C[i][j] = 0; // Replace even element with 0
            } else {
                local_C[i][j] = 1; // Replace odd element with 1
            }
        }
    }
}

void print_matrix(const char *name, int matrix[N][N]) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if number of processes is appropriate for the matrix size
    if (size < 2) {
        fprintf(stderr, "This program requires at least 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int A[N][N];
    int B[N][N];
    int local_A[N][N];
    int local_B[N][N];
    int local_C[N][N];

    // Initialize matrices A and B with random values
    srand(1234);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10; // Random value between 0 and 9
            B[i][j] = rand() % 10; // Random value between 0 and 9
        }
    }

    // Print original matrices A and B
    if (rank == 0) {
        print_matrix("A", A);
        printf("\n");
        print_matrix("B", B);
        printf("\n");
    }

    // Scatter matrices A and B to all processes
    MPI_Scatter(A, N * N / size, MPI_INT, local_A, N * N / size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, N * N / size, MPI_INT, local_B, N * N / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local matrix addition
    add_matrices(local_A, local_B, local_C, N);

    // Gather the results of local_C back to matrix C at root process (rank 0)
    MPI_Gather(local_C, N * N / size, MPI_INT, A, N * N / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Apply element-wise transformation to matrix C at root process (rank 0)
    if (rank == 0) {
        // Display the resulting matrix C before transformation
        printf("Resulting Matrix C (before transformation):\n");
        print_matrix("C", A);

        // Apply element-wise transformation to matrix C
        apply_element_transformation(A, N);

        // Display the resulting matrix C after transformation
        printf("\nResulting Matrix C (after transformation):\n");
        print_matrix("C", A);
    }

    MPI_Finalize();
    return 0;
}

