#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Structure to hold random number generator state
typedef struct {
    long ma[55];
    int inext, inextp, iff;
} RNGState;

double ran3(int *idum, RNGState *state);

int main() {
    int n = 60000; // Reduced size for testing (adjust as needed)
    int i;
    double **h;

    // Allocate memory with error checking
    h = (double **)malloc(n * sizeof(double *));
    if (h == NULL) {
        fprintf(stderr, "Failed to allocate row pointers\n");
        return 1;
    }
    for (i = 0; i < n; i++) {
        h[i] = (double *)malloc(n * sizeof(double));
        if (h[i] == NULL) {
            fprintf(stderr, "Failed to allocate row %d\n", i);
            for (int j = 0; j < i; j++) free(h[j]);
            free(h);
            return 1;
        }
    }

    // Parallel region
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int iseed = -(99 + thread_id);
        RNGState state = {{0}, 0, 0, 0}; // Thread-local RNG state

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = ran3(&iseed, &state);
                h[i][j] = val;
                h[j][i] = val; // Symmetry (j <= i < n, so no bounds check needed)
            }
        }
    }
    printf("Matrix created successfully with %d threads (n=%d)!\n", 
           omp_get_max_threads(), n);

    // Free memory
    for (i = 0; i < n; i++) {
        free(h[i]);
    }
    free(h);

    return 0;
}

double ran3(int *idum, RNGState *state) {
    const int mbig = 1000000000, mseed = 161803398, mz = 0;
    const double fac = 1.0 / 1000000000.0;
    long mj, mk;
    int i, ii, k;

    if (*idum < 0 || state->iff == 0) {
        state->iff = 1;
        mj = mseed - (*idum < 0 ? -*idum : *idum);
        mj %= mbig;
        state->ma[54] = mj;
        mk = 1;
        for (i = 1; i <= 54; i++) {
            ii = (21 * i) % 55;
            state->ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < mz) mk += mbig;
            mj = state->ma[ii - 1];
        }
        for (k = 1; k <= 4; k++) {
            for (i = 0; i < 55; i++) {
                state->ma[i] -= state->ma[(i + 30) % 55];
                if (state->ma[i] < mz) state->ma[i] += mbig;
            }
        }
        state->inext = 0;
        state->inextp = 31;
        *idum = 1;
    }

    if (++state->inext == 55) state->inext = 0;
    if (++state->inextp == 55) state->inextp = 0;
    mj = state->ma[state->inext] - state->ma[state->inextp];
    if (mj < mz) mj += mbig;
    state->ma[state->inext] = mj;
    return mj * fac;
}
