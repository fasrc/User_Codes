#include <iostream>
#include <vector>
#include <omp.h>

// Structure to hold random number generator state
struct RNGState {
    std::vector<long> ma;
    int inext, inextp, iff;
    RNGState() : ma(55, 0), inext(0), inextp(0), iff(0) {}
};

// Function prototype for the random number generator
double ran3(int& idum, RNGState& state);

int main() {
    const int n = 60000; // Matrix dimension

    // Allocate memory for the matrix using std::vector
    std::vector<std::vector<double>> h(n, std::vector<double>(n));

    // Create random symmetric test matrix with OpenMP
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int iseed = -(99 + thread_id); // Unique seed per thread
        RNGState state; // Thread-local RNG state

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = ran3(iseed, state);
                h[i][j] = val;
                h[j][i] = val; // Symmetric matrix
            }
        }
    }
    std::cout << "Hamiltonian matrix created successfully with " 
              << omp_get_max_threads() << " threads (n=" << n << ")!" 
              << std::endl;

    // No manual memory freeing needed; std::vector handles it

    return 0;
}

// Random number generator (ran3)
double ran3(int& idum, RNGState& state) {
    const int mbig = 1000000000, mseed = 161803398, mz = 0;
    const double fac = 1.0 / mbig;
    long mj, mk;
    int i, ii, k;

    if (idum < 0 || state.iff == 0) {
        state.iff = 1;
        mj = mseed - (idum < 0 ? -idum : idum);
        mj %= mbig;
        state.ma[54] = mj;
        mk = 1;
        for (i = 1; i <= 54; i++) {
            ii = (21 * i) % 55;
            state.ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < mz) mk += mbig;
            mj = state.ma[ii - 1];
        }
        for (k = 1; k <= 4; k++) {
            for (i = 0; i < 55; i++) {
                state.ma[i] -= state.ma[(i + 30) % 55];
                if (state.ma[i] < mz) state.ma[i] += mbig;
            }
        }
        state.inext = 0;
        state.inextp = 31;
        idum = 1;
    }

    if (++state.inext == 55) state.inext = 0;
    if (++state.inextp == 55) state.inextp = 0;
    mj = state.ma[state.inext] - state.ma[state.inextp];
    if (mj < mz) mj += mbig;
    state.ma[state.inext] = mj;
    return mj * fac;
}
