#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <openblas/cblas.h>

#define EPS 1e-9

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_rand(double *X, int n) {
    for (int i = 0; i < n*n; i++) X[i] = (double)(rand() % 10);
}

static void zero_m(double *X, int n) {
    for (int i = 0; i < n*n; i++) X[i] = 0.0;
}

static int verify(const double* C1, const double* C2, int n) {
    int size = n*n;
    for (int i = 0; i < size; i++) {
        if (fabs(C1[i] - C2[i]) >= EPS) return 0;
    }
    return 1;
}

static void naive_mm(const double *A, const double *B, double *C, int n) {
    zero_m(C, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

/* GEMM */
static void gemm_blas(const double *A, const double *B, double *C, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0, A, n,
                     B, n,
                0.0, C, n);
}

/* helpers for Strassen */
static void add_m(const double *X, const double *Y, double *Z, int n) {
    for (int i = 0; i < n*n; i++) Z[i] = X[i] + Y[i];
}
static void sub_m(const double *X, const double *Y, double *Z, int n) {
    for (int i = 0; i < n*n; i++) Z[i] = X[i] - Y[i];
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "malloc failed (%zu bytes)\n", bytes);
        exit(1);
    }
    return p;
}

/*Strassen Recursion.*/
static void strassen_rec(const double *A, const double *B, double *C, int n, int cutoff) {
    if (n <= cutoff) {
        gemm_blas(A, B, C, n);
        return;
    }

    int h = n / 2;
    int sz = h * h;

    double *A11 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *A12 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *A21 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *A22 = (double*)xmalloc((size_t)sz*sizeof(double));

    double *B11 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *B12 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *B21 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *B22 = (double*)xmalloc((size_t)sz*sizeof(double));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            A11[i*h+j] = A[i*n + j];
            A12[i*h+j] = A[i*n + (j+h)];
            A21[i*h+j] = A[(i+h)*n + j];
            A22[i*h+j] = A[(i+h)*n + (j+h)];

            B11[i*h+j] = B[i*n + j];
            B12[i*h+j] = B[i*n + (j+h)];
            B21[i*h+j] = B[(i+h)*n + j];
            B22[i*h+j] = B[(i+h)*n + (j+h)];
        }
    }

    double *P1 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *P2 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *P3 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *P4 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *P5 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *P6 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *P7 = (double*)xmalloc((size_t)sz*sizeof(double));

    double *T1 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *T2 = (double*)xmalloc((size_t)sz*sizeof(double));

    add_m(A11, A22, T1, h);
    add_m(B11, B22, T2, h);
    strassen_rec(T1, T2, P1, h, cutoff);

    add_m(A21, A22, T1, h);
    strassen_rec(T1, B11, P2, h, cutoff);

    sub_m(B12, B22, T2, h);
    strassen_rec(A11, T2, P3, h, cutoff);
    
    sub_m(B21, B11, T2, h);
    strassen_rec(A22, T2, P4, h, cutoff);

    add_m(A11, A12, T1, h);
    strassen_rec(T1, B22, P5, h, cutoff);

    sub_m(A21, A11, T1, h);
    add_m(B11, B12, T2, h);
    strassen_rec(T1, T2, P6, h, cutoff);

    sub_m(A12, A22, T1, h);
    add_m(B21, B22, T2, h);
    strassen_rec(T1, T2, P7, h, cutoff);

    double *C11 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *C12 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *C21 = (double*)xmalloc((size_t)sz*sizeof(double));
    double *C22 = (double*)xmalloc((size_t)sz*sizeof(double));

    add_m(P1, P4, T1, h);
    sub_m(T1, P5, T2, h);
    add_m(T2, P7, C11, h);

    add_m(P3, P5, C12, h);

    add_m(P2, P4, C21, h);

    sub_m(P1, P2, T1, h);
    add_m(T1, P3, T2, h);
    add_m(T2, P6, C22, h);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            C[i*n + j]         = C11[i*h + j];
            C[i*n + (j+h)]     = C12[i*h + j];
            C[(i+h)*n + j]     = C21[i*h + j];
            C[(i+h)*n + (j+h)] = C22[i*h + j];
        }
    }

    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(P1);  free(P2);  free(P3);  free(P4);  free(P5);  free(P6);  free(P7);
    free(T1);  free(T2);
    free(C11); free(C12); free(C21); free(C22);
}

static void strassen_mm(const double *A, const double *B, double *C, int n) {
    const int cutoff = 128; 
    strassen_rec(A, B, C, n, cutoff);
}

int main(void) {
    int sizes[] = {1024, 2048, 4096, 8192};
    int count = (int)(sizeof(sizes)/sizeof(sizes[0]));

    srand((unsigned)time(NULL));

    printf("Order, GEMM(s), Strassen(s), CW_estimate(s)\n");

    double gemm_1024 = 0.0;

    for (int i = 0; i < count; i++) {
        int n = sizes[i];

        double *A    = (double*)malloc((size_t)n*n*sizeof(double));
        double *B    = (double*)malloc((size_t)n*n*sizeof(double));
        double *Cref = (double*)malloc((size_t)n*n*sizeof(double));
        double *Cs   = (double*)malloc((size_t)n*n*sizeof(double));
        double *Cg   = (double*)malloc((size_t)n*n*sizeof(double));

        if (!A || !B || !Cref || !Cs || !Cg) {
            fprintf(stderr, "Memory allocation failed at n=%d\n", n);
            free(A); free(B); free(Cref); free(Cs); free(Cg);
            return 1;
        }

        fill_rand(A, n);
        fill_rand(B, n);

        if (n == 1024) naive_mm(A, B, Cref, n);
        else          gemm_blas(A, B, Cref, n);

        /* Strassen (skips 8192 - memory constraints) */
        double stras_time = -1.0;
        if (n != 8192) {
            double t1 = now_sec();
            strassen_mm(A, B, Cs, n);
            double t2 = now_sec();
            stras_time = t2 - t1;

            int sV = verify(Cref, Cs, n);
            if (!sV) printf("WARNING: Strassen verify FAIL at n=%d\n", n);
        }

        /* GEMM timing */
        double t3 = now_sec();
        gemm_blas(A, B, Cg, n);
        double t4 = now_sec();
        double gemm_time = t4 - t3;

        int gV = verify(Cref, Cg, n);
        if (!gV) printf("WARNING: GEMM verify FAIL at n=%d\n", n);

        if (n == 1024) gemm_1024 = gemm_time;

        /* CW estimate*/
        double cw_est = 0.0;
        if (gemm_1024 > 0.0) {
            cw_est = gemm_1024 * pow((double)n / 1024.0, 2.376);
        }

        if (stras_time < 0.0) {
            printf("%d, %.6f, N/A, %.6f\n", n, gemm_time, cw_est);
        } else {
            printf("%d, %.6f, %.6f, %.6f\n", n, gemm_time, stras_time, cw_est);
        }

        free(A); free(B); free(Cref); free(Cs); free(Cg);
    }

    return 0;
}