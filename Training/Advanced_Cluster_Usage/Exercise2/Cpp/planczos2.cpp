#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

// Global variables (replacing Fortran modules)
int niter;           // Number of Lanczos iterations
int nkeep;           // Number of eigenvalues to keep
std::fstream lvec_file; // File stream for Lanczos vectors
bool reorthog;       // Flag for reorthogonalization
bool writetodisk;    // Flag to write vectors to disk
std::vector<double> lvec; // Vector for Lanczos vectors if not writetodisk
double timeorthog;   // Timing for orthogonalization
double timelast_ort; // Last orthogonalization time

// Function prototypes
void open_file();
void close_file();
void write_file(int iter, const std::vector<double>& v, int n);
void read_file(int jvec, std::vector<double>& temp_v, int n);
void applyh(int n, const std::vector<double>& h, const std::vector<double>& vecin, std::vector<double>& vecout);
void dnormvec(int n, std::vector<double>& dvec, double& dnorm);
void dvecproj(int n, std::vector<double>& dvec1, const std::vector<double>& dvec2, double& dsclrprod);
void reorthogonalize(std::vector<double>& w, std::vector<double>& v, int n, int iter);
// External functions (assumed defined elsewhere)
extern "C" void tqli(int nm, int n, double* d, double* e, double* z, int* ierr);
extern "C" void eigsrt(double* d, double* v, int n, int np);

int main() {
    // Initialize flags
    writetodisk = true;
    reorthog = true;

    // Open file for vectors if writing to disk
    if (writetodisk) open_file();

    // Run parameters
    int n = 30000;  // Matrix dimension
    nkeep = 5;      // Number of eigenvalues to keep
    niter = 50;     // Number of iterations

    // Allocate memory using vectors
    std::vector<double> h(n * n);      // Random symmetric matrix
    std::vector<double> dh(n);         // Unused in this code
    std::vector<double> eh(n);         // Unused in this code
    std::vector<double> zh(n * n);     // Unused in this code
    std::vector<double> v(n);          // Current Lanczos vector
    std::vector<double> w(n);          // Temporary vector
    std::vector<double> alpha(niter);  // Diagonal coefficients
    std::vector<double> beta(niter);   // Off-diagonal coefficients
    std::vector<double> d(niter);      // Tridiagonal diagonal
    std::vector<double> e(niter);      // Tridiagonal off-diagonal
    std::vector<double> z(n * niter);  // Eigenvectors/transforms

    // Allocate lvec if not writing to disk
    if (!writetodisk) {
        lvec.resize(n * niter);
    }

    // Seed random number generator
    std::mt19937 rng(99);  // Mersenne Twister with seed 99
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Create random symmetric matrix h
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            h[j * n + i] = dist(rng);
            h[i * n + j] = h[j * n + i];  // Symmetry
        }
    }

    // Initialize starting vector v
    double dnorm_val = 1.0 / std::sqrt(static_cast<double>(n));
    for (int i = 0; i < n; i++) {
        v[i] = dnorm_val;
    }
    double da;
    dnormvec(n, v, da);  // Normalize v

    // Lanczos iterations
    int iter = 0;
    while (iter < niter) {
        iter++;
        write_file(iter, v, n);        // Write current vector
        applyh(n, h, v, w);            // w = H * v
        dvecproj(n, w, v, da);         // Project w onto v
        alpha[iter - 1] = da;          // Store alpha

        if (reorthog) {
            reorthogonalize(w, v, n, iter);  // Reorthogonalize w
        }

        if (iter < niter) {
            double db;
            dnormvec(n, w, db);        // Normalize w
            beta[iter] = db;           // Store beta (beta[1] unused)
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                v[i] = w[i];           // Update v for next iteration
            }
        }

        // Prepare tridiagonal matrix for diagonalization
        for (int j = 0; j < niter; j++) {
            d[j] = 0.0;
            e[j] = 0.0;
        }
        for (int j = 0; j < iter; j++) {
            d[j] = alpha[j];
            if (j < iter) e[j] = beta[j];  // Note: e(1)=beta(1), etc.
        }
        // Initialize z as identity matrix (n x iter portion)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < iter; j++) {
                z[j * n + i] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Diagonalize tridiagonal matrix
        int ierr = 0;
        tqli(n, iter, d.data(), e.data(), z.data(), &ierr);
        eigsrt(d.data(), z.data(), iter, n);  // Sort eigenvalues and vectors

        if (ierr != 0) {
            std::cout << "diag ierr=" << ierr << std::endl;
        }

        // Print results
        std::cout << nkeep << " lowest eigenvalues - Lanczos" << std::endl;
        std::cout << "iteration: " << iter << std::endl;
        for (int i = 0; i < nkeep; i++) {
            std::cout << i + 1 << " " << d[i] << std::endl;
        }
    }

    std::cout << "Lanczos iterations finished..." << std::endl;
    if (writetodisk) close_file();

    return 0;
}

// File I/O functions
void open_file() {
    lvec_file.open("lanczosvector.lvec", std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!lvec_file.is_open()) {
        std::cerr << "Error opening lanczosvector.lvec" << std::endl;
        std::exit(1);
    }
}

void close_file() {
    if (lvec_file.is_open()) {
        lvec_file.close();
    }
}

void write_file(int iter, const std::vector<double>& v, int n) {
    if (iter == 1) {
        lvec_file.seekp(0, std::ios::beg);  // Rewind at first write
    }
    lvec_file.write(reinterpret_cast<const char*>(v.data()), n * sizeof(double));
    if (!lvec_file) {
        std::cerr << "Error writing vector " << iter << std::endl;
        std::exit(1);
    }
}

void read_file(int jvec, std::vector<double>& temp_v, int n) {
    if (jvec == 1) {
        lvec_file.seekg(0, std::ios::beg);  // Rewind at first read
    }
    lvec_file.read(reinterpret_cast<char*>(temp_v.data()), n * sizeof(double));
    if (!lvec_file) {
        std::cerr << "Error reading vector " << jvec << std::endl;
        std::exit(1);
    }
}

// Matrix-vector and vector operations
void applyh(int n, const std::vector<double>& h, const std::vector<double>& vecin, std::vector<double>& vecout) {
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        vecout[j] = 0.0;
        for (int i = 0; i < n; i++) {
            vecout[j] += h[i * n + j] * vecin[i];
        }
    }
}

void dnormvec(int n, std::vector<double>& dvec, double& dnorm) {
    double temp = 0.0;
    #pragma omp parallel for reduction(+:temp)
    for (int i = 0; i < n; i++) {
        temp += dvec[i] * dvec[i];
    }
    dnorm = std::sqrt(temp);
    double scale = 1.0 / dnorm;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dvec[i] *= scale;
    }
}

void dvecproj(int n, std::vector<double>& dvec1, const std::vector<double>& dvec2, double& dsclrprod) {
    double temp = 0.0;
    #pragma omp parallel for reduction(+:temp)
    for (int i = 0; i < n; i++) {
        temp += dvec1[i] * dvec2[i];
    }
    dsclrprod = temp;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dvec1[i] -= dvec2[i] * dsclrprod;
    }
}

void reorthogonalize(std::vector<double>& w, std::vector<double>& v, int n, int iter) {
    std::vector<double> temp_v(n);
    int nread = iter;
    for (int j = 0; j < nread; j++) {
        int jvec = j + 1;  // Adjust for 1-based indexing in file
        read_file(jvec, temp_v, n);
        double dsclrprod = 0.0;
        #pragma omp parallel for reduction(+:dsclrprod)
        for (int i = 0; i < n; i++) {
            dsclrprod += w[i] * temp_v[i];
        }
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            w[i] -= temp_v[i] * dsclrprod;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        v[i] = w[i];
    }
}

