#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cstring>
#include <omp.h>

#define at(y, x, mat) (mat[y * n + x])

void matrix_multiplication(int n, int *a, int *b, int *c) {
    int i, j, k;
    #pragma omp parallel for private(i, j, k) shared(a, b, c)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            int t = 0;
            for (k = 0; k < n; k++) {
                t += at(i, k, a) * at(k, j, b);
            }
            at(i, j, c) = t;
        }
    }
}

void print_matrix(int n, int *mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", at(i, j, mat));
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int n;
    if (argc != 3 || strcmp(argv[1], "-n") != 0) {
        printf("Usage: %s -n <matrix_size>\n", argv[0]);
        return 1;
    }
    n = atoi(argv[2]); 
    int a[n*n];
    int b[n*n];
    int c[n*n];

    for (int i = 0; i < n*n; i++) {
        a[i] = i % 10; // Fill matrix 'a' with values 0 to 9 in a cyclic pattern
        b[i] = (i % 10) + 1; // Fill matrix 'b' with values 1 to 10 in a cyclic pattern
    }

    double start_time = omp_get_wtime();

    matrix_multiplication(n, a, b, c);

    double end_time = omp_get_wtime();

    double elapsed_time = end_time - start_time;
    
    // printf("Number of OpenMP threads used: %d\n", omp_get_max_threads());
    // printf("Matrix multiplication took %.4f seconds.\n", elapsed_time);
    printf("%.4f, ", elapsed_time);
    // printf("Matrix A:\n");
    // print_matrix(n, a);
    // printf("\nMatrix B:\n");
    // print_matrix(n, b);
    // printf("\nMatrix C (Result of A * B):\n");
    // print_matrix(n, c);

    return 0;
}
