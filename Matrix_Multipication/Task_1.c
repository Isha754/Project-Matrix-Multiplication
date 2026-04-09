#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int M, N, K;

    printf("Enter M, N, K (512 ~ 2048): ");
    if (scanf("%d %d %d", &M, &N, &K) != 3) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    if (M < 512 || M > 2048 || N < 512 || N > 2048 || K < 512 || K > 2048) {
        fprintf(stderr, "M, N, K must be in [512, 2048].\n");
        return 1;
    }

    size_t sizeA = (size_t)M * (size_t)N;
    size_t sizeB = (size_t)N * (size_t)K;
    size_t sizeC = (size_t)M * (size_t)K;

    double *A = (double*)malloc(sizeA * sizeof(double));
    double *B = (double*)malloc(sizeB * sizeof(double));
    double *C = (double*)calloc(sizeC, sizeof(double)); // sets to 0

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(A); free(B); free(C);
        return 1;
    }

    srand((unsigned)time(NULL));
    for (size_t i = 0; i < sizeA; i++) A[i] = (double)(rand() % 10);
    for (size_t i = 0; i < sizeB; i++) B[i] = (double)(rand() % 10);

    clock_t start_time = clock();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }

    clock_t end_time = clock();

    printf("Matrix multiplication time: %.6f seconds\n",
           (double)(end_time - start_time) / (double)CLOCKS_PER_SEC);

    free(A);
    free(B);
    free(C);
    return 0;
}