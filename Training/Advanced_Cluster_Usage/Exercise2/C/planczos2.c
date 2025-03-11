#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

// Global variables (replacing Fortran modules)
int niter;           // Number of Lanczos iterations
int nkeep;           // Number of eigenvalues to keep
FILE *lvec_file;     // File handle for Lanczos vectors
bool reorthog;       // Flag for reorthogonalization
bool writetodisk;    // Flag to write vectors to disk
double *lvec;        // Array for Lanczos vectors if not writetodisk
double timeorthog;   // Timing for orthogonalization
double timelast_ort; // Last orthogonalization time

// Function prototypes
void open_file();
void close_file();
void write_file(int iter, double *v, int n);
void read_file(int jvec, double *temp_v, int n);
void applyh(int n, double *h, double *vecin, double *vecout);
void dnormvec(int n, double *dvec, double *dnorm);
void dvecproj(int n, double *dvec1, double *dvec2, double *dsclrprod);
void reorthogonalize(double *w, double *v, int n, int iter);
// External functions (assumed defined elsewhere)
void tqli(int nm, int n, double *d, double *e, double *z, int *ierr);
void eigsrt(double *d, double *v, int n, int np);

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

    // Allocate memory
    double *h = malloc(n * n * sizeof(double));      // Random symmetric matrix
    double *dh = malloc(n * sizeof(double));         // Unused in this code
    double *eh = malloc(n * sizeof(double));         // Unused in this code
    double *zh = malloc(n * n * sizeof(double));     // Unused in this code
    double *v = malloc(n * sizeof(double));          // Current Lanczos vector
    double *w = malloc(n * sizeof(double));          // Temporary vector
    double *alpha = malloc(niter * sizeof(double));  // Diagonal coefficients
    double *beta = malloc(niter * sizeof(double));   // Off-diagonal coefficients
    double *d = malloc(niter * sizeof(double));      // Tridiagonal diagonal
    double *e = malloc(niter * sizeof(double));      // Tridiagonal off-diagonal
    double *z = malloc(n * niter * sizeof(double));  // Eigenvectors/transforms

    // Check allocations (basic error handling)
    if (!h || !dh || !eh || !zh || !v || !w || !alpha || !beta || !d || !e || !z) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Allocate lvec if not writing to disk
    if (!writetodisk) {
        lvec = malloc(n * niter * sizeof(double));
        if (!lvec) {
            fprintf(stderr, "Memory allocation failed for lvec\n");
            exit(1);
        }
    }

    // Seed random number generator
    srand48(99);  // Positive seed for drand48()

    // Create random symmetric matrix h
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            h[j * n + i] = drand48();
            h[i * n + j] = h[j * n + i];  // Symmetry
        }
    }

    // Initialize starting vector v
    double dnorm_val = 1.0 / sqrt((double)n);
    for (int i = 0; i < n; i++) {
        v[i] = dnorm_val;
    }
    double da;
    dnormvec(n, v, &da);  // Normalize v

    // Lanczos iterations
    int iter = 0;
    while (iter < niter) {
        iter++;
        write_file(iter, v, n);        // Write current vector
        applyh(n, h, v, w);            // w = H * v
        dvecproj(n, w, v, &da);        // Project w onto v
        alpha[iter - 1] = da;          // Store alpha

        if (reorthog) {
            reorthogonalize(w, v, n, iter);  // Reorthogonalize w
        }

        if (iter < niter) {
            double db;
            dnormvec(n, w, &db);       // Normalize w
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
        tqli(n, iter, d, e, z, &ierr);
        eigsrt(d, z, iter, n);  // Sort eigenvalues and vectors

        if (ierr != 0) {
            printf("diag ierr=%d\n", ierr);
        }

        // Print results
        printf("%d lowest eigenvalues - Lanczos\n", nkeep);
        printf("iteration: %d\n", iter);
        for (int i = 0; i < nkeep; i++) {
            printf("%d %f\n", i + 1, d[i]);
        }
    }

    printf("Lanczos iterations finished...\n");
    if (writetodisk) close_file();

    // Free allocated memory
    free(h);
    free(dh);
    free(eh);
    free(zh);
    free(v);
    free(w);
    free(alpha);
    free(beta);
    free(d);
    free(e);
    free(z);
    if (!writetodisk) free(lvec);

    return 0;
}

// File I/O functions
void open_file() {
    lvec_file = fopen("lanczosvector.lvec", "wb+");
    if (!lvec_file) {
        fprintf(stderr, "Error opening lanczosvector.lvec\n");
        exit(1);
    }
}

void close_file() {
    if (lvec_file) {
        fclose(lvec_file);
        lvec_file = NULL;
    }
}

void write_file(int iter, double *v, int n) {
    if (iter == 1) {
        fseek(lvec_file, 0, SEEK_SET);  // Rewind at first write
    }
    size_t written = fwrite(v, sizeof(double), n, lvec_file);
    if (written != (size_t)n) {
        fprintf(stderr, "Error writing vector %d\n", iter);
        exit(1);
    }
}

void read_file(int jvec, double *temp_v, int n) {
    if (jvec == 1) {
        fseek(lvec_file, 0, SEEK_SET);  // Rewind at first read
    }
    size_t read_count = fread(temp_v, sizeof(double), n, lvec_file);
    if (read_count != (size_t)n) {
        fprintf(stderr, "Error reading vector %d\n", jvec);
        exit(1);
    }
}

// Matrix-vector and vector operations
void applyh(int n, double *h, double *vecin, double *vecout) {
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        vecout[j] = 0.0;
        for (int i = 0; i < n; i++) {
            vecout[j] += h[i * n + j] * vecin[i];
        }
    }
}

void dnormvec(int n, double *dvec, double *dnorm) {
    double temp = 0.0;
    #pragma omp parallel for reduction(+:temp)
    for (int i = 0; i < n; i++) {
        temp += dvec[i] * dvec[i];
    }
    *dnorm = sqrt(temp);
    double scale = 1.0 / *dnorm;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dvec[i] *= scale;
    }
}

void dvecproj(int n, double *dvec1, double *dvec2, double *dsclrprod) {
    double temp = 0.0;
    #pragma omp parallel for reduction(+:temp)
    for (int i = 0; i < n; i++) {
        temp += dvec1[i] * dvec2[i];
    }
    *dsclrprod = temp;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dvec1[i] -= dvec2[i] * (*dsclrprod);
    }
}

void reorthogonalize(double *w, double *v, int n, int iter) {
    double *temp_v = malloc(n * sizeof(double));
    if (!temp_v) {
        fprintf(stderr, "Memory allocation failed in reorthogonalize\n");
        exit(1);
    }

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

    free(temp_v);
}

