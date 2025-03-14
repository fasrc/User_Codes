#include <stdio.h>
#include <stdlib.h>

// Function prototype for the random number generator
double ran3(int *idum);

int main() {
    int n = 60000; // Matrix dimension
    int i, j;
    int iseed = -99;
    double **h;

    // Allocate memory for the matrix
    h = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++) {
        h[i] = (double *)malloc(n * sizeof(double));
    }

    // Create random symmetric test matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            h[i][j] = ran3(&iseed);
            h[j][i] = h[i][j]; // Symmetric matrix
        }
    }
    printf("Hamiltonian matrix created successfully!\n");

    // Free memory
    for (i = 0; i < n; i++) {
        free(h[i]);
    }
    free(h);

    return 0;
}

// Random number generator (ran3)
double ran3(int *idum) {
    static int iff = 0;
    static int inext, inextp;
    static long ma[55];
    static int mbig = 1000000000, mseed = 161803398, mz = 0;
    static double fac = 1.0 / 1000000000.0;
    long mj, mk;
    int i, ii, k;

    if (*idum < 0 || iff == 0) {
        iff = 1;
        mj = mseed - (*idum < 0 ? -*idum : *idum);
        mj %= mbig;
        ma[54] = mj;
        mk = 1;
        for (i = 1; i <= 54; i++) {
            ii = (21 * i) % 55;
            ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < mz) mk += mbig;
            mj = ma[ii - 1];
        }
        for (k = 1; k <= 4; k++) {
            for (i = 0; i < 55; i++) {
                ma[i] -= ma[(i + 30) % 55];
                if (ma[i] < mz) ma[i] += mbig;
            }
        }
        inext = 0;
        inextp = 31;
        *idum = 1;
    }

    if (++inext == 55) inext = 0;
    if (++inextp == 55) inextp = 0;
    mj = ma[inext] - ma[inextp];
    if (mj < mz) mj += mbig;
    ma[inext] = mj;
    return mj * fac;
}

