#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // For std::max, std::min

extern "C" void tqli(int nm, int n, double* d, double* e, double* z, int* ierr);
extern "C" void eigsrt(double* d, double* v, int n, int np);
double pythag(double a, double b); // Required by tqli

// Function declarations
void tred2(int nm, int n, std::vector<double>& a, std::vector<double>& d, std::vector<double>& e, std::vector<double>& z);
double ran3(int& idum);
void shellsort(int n, int ind, int dim1, std::vector<int>& arr);
void piksrt(int n, std::vector<double>& arr);


void tred2(int nm, int n, std::vector<double>& a, std::vector<double>& d, std::vector<double>& e, std::vector<double>& z) {
    double f, g, h, hh, scale;  // Declare all variables at function scope

    // Copy lower triangle of A to Z and initialize D
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            z[j * nm + i] = a[j * nm + i];
        }
        d[i] = a[(n - 1) * nm + i];
    }

    // Handle the special case of n = 1
    if (n == 1) {
        e[0] = 0.0;
        z[0] = 1.0;
        return;
    }

    // Tridiagonalization loop
    for (int ii = 1; ii < n; ii++) {
        int i = n - 1 - (ii - 1); // Adjusted for 0-based indexing
        int l = i - 1;
        h = 0.0;
        scale = 0.0;

        // Compute scale factor
        if (l >= 0) { // Adjusted for 0-based indexing
            for (int k = 0; k <= l; k++) {
                scale += std::abs(d[k]);
            }
        }

        if (scale == 0.0) {
            e[i] = d[l];
            for (int j = 0; j <= l; j++) {
                d[j] = z[l * nm + j];
                z[i * nm + j] = 0.0;
                z[j * nm + i] = 0.0;
            }
        } else {
            // Scale the vector and compute h
            for (int k = 0; k <= l; k++) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            f = d[l];
            g = -std::copysign(std::sqrt(h), f);
            e[i] = scale * g;
            h -= f * g;
            d[l] = f - g;

            // Form A*U
            for (int j = 0; j <= l; j++) {
                e[j] = 0.0;
            }
            for (int j = 0; j <= l; j++) {
                f = d[j];
                z[j * nm + i] = f;
                g = e[j] + z[j * nm + j] * f;
                int jp1 = j + 1;
                if (jp1 <= l) {
                    for (int k = jp1; k <= l; k++) {
                        g += z[k * nm + j] * d[k];
                        e[k] += z[k * nm + j] * f;
                    }
                }
                e[j] = g;
            }

            // Form P and Q
            f = 0.0;
            for (int j = 0; j <= l; j++) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            hh = f / (h + h);
            for (int j = 0; j <= l; j++) {
                e[j] -= hh * d[j];
            }

            // Reduce A
            for (int j = 0; j <= l; j++) {
                f = d[j];
                g = e[j];
                for (int k = j; k <= l; k++) {
                    z[k * nm + j] -= f * e[k] + g * d[k];
                }
                d[j] = z[l * nm + j];
                z[i * nm + j] = 0.0;
            }
        }
        d[i] = h;
    }

    // Accumulate transformation matrices
    for (int i = 1; i < n; i++) {
        int l = i - 1;
        z[(n - 1) * nm + l] = z[l * nm + l];
        z[l * nm + l] = 1.0;
        h = d[i];  // Now h is in scope
        if (h != 0.0) {
            for (int k = 0; k <= l; k++) {
                d[k] = z[k * nm + i] / h;
            }
            for (int j = 0; j <= l; j++) {
                g = 0.0;  // Now g is in scope
                for (int k = 0; k <= l; k++) {
                    g += z[k * nm + i] * z[k * nm + j];
                }
                for (int k = 0; k <= l; k++) {
                    z[k * nm + j] -= g * d[k];
                }
            }
        }
        for (int k = 0; k <= l; k++) {
            z[k * nm + i] = 0.0;
        }
    }

    // Finalize output
    for (int i = 0; i < n; i++) {
        d[i] = z[(n - 1) * nm + i];
        z[(n - 1) * nm + i] = 0.0;
    }
    z[(n - 1) * nm + (n - 1)] = 1.0;
    e[0] = 0.0;
}


double pythag(double a, double b) {
    double p = std::max(std::abs(a), std::abs(b));
    if (p == 0.0) {
        return 0.0;
    }
    double r = std::pow(std::min(std::abs(a), std::abs(b)) / p, 2);
    while (true) {
        double t = 4.0 + r;
        if (t == 4.0) break;
        double s = r / t;
        double u = 1.0 + 2.0 * s;
        p *= u;
        r = std::pow(s / u, 2) * r;
    }
    return p;
}

extern "C" void tqli(int nm, int n, double* d, double* e, double* z, int* ierr) {
    *ierr = 0;
    if (n == 1) return;

    // Shift subdiagonal elements
    for (int i = 1; i < n; i++) {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    double f = 0.0;
    double tst1 = 0.0;

    // Main QL iteration
    for (int l = 0; l < n; l++) {
        int j = 0;
        double h = std::abs(d[l]) + std::abs(e[l]);
        if (tst1 < h) tst1 = h;
        int m = l;
        while (m < n) {
            double tst2 = tst1 + std::abs(e[m]);
            if (tst2 == tst1) break;
            m++;
        }
        if (m == l) {
            d[l] += f;
            continue;
        }
        do {
            if (j == 30) {
                *ierr = l + 1; // Adjusted for 1-based error reporting
                return;
            }
            j++;
            int l1 = l + 1;
            int l2 = l1 + 1;
            double g = d[l];
            double p = (d[l1] - g) / (2.0 * e[l]);
            double r = pythag(p, 1.0);
            d[l] = e[l] / (p + std::copysign(r, p));
            d[l1] = e[l] * (p + std::copysign(r, p));
            double dl1 = d[l1];
            h = g - d[l];
            if (l2 < n) {
                for (int i = l2; i < n; i++) {
                    d[i] -= h;
                }
            }
            f += h;

            // QL transformation
            p = d[m];
            double c = 1.0;
            double c2 = c;
            double el1 = e[l1];
            double s = 0.0;
            int mml = m - l;
            double c3, s2; // Declared outside loop for later use
            for (int ii = 0; ii < mml; ii++) {
                c3 = c2;
                c2 = c;
                s2 = s;
                int i = m - 1 - ii;
                g = c * e[i];
                h = c * p;
                r = pythag(p, e[i]);
                e[i + 1] = s * r;
                s = e[i] / r;
                c = p / r;
                p = c * d[i] - s * g;
                d[i + 1] = h + s * (c * g + s * d[i]);
                for (int k = 0; k < n; k++) {
                    h = z[(i + 1) * nm + k];
                    z[(i + 1) * nm + k] = s * z[i * nm + k] + c * h;
                    z[i * nm + k] = c * z[i * nm + k] - s * h;
                }
            }
            p = -s * s2 * c3 * el1 * e[l] / dl1;
            e[l] = s * p;
            d[l] = c * p;
            double tst2 = tst1 + std::abs(e[l]);
            if (tst2 <= tst1) break;
        } while (true);
        d[l] += f;
    }

    // Order eigenvalues and eigenvectors
    for (int ii = 1; ii < n; ii++) {
        int i = ii - 1;
        int k = i;
        double p = d[i];
        for (int j = ii; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (int j = 0; j < n; j++) {
                p = z[i * nm + j];
                z[i * nm + j] = z[k * nm + j];
                z[k * nm + j] = p;
            }
        }
    }
}

extern "C" void eigsrt(double* d, double* v, int n, int np) {
    for (int i = 0; i < n - 1; i++) {
        int k = i;
        double p = d[i];
        for (int j = i + 1; j < n; j++) {
            if (d[j] >= p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (int j = 0; j < n; j++) {
                p = v[i * np + j];
                v[i * np + j] = v[k * np + j];
                v[k * np + j] = p;
            }
        }
    }
}

double ran3(int& idum) {
    const int MBIG = 1000000000;
    const int MSEED = 161803398;
    const int MZ = 0;
    const double FAC = 1.0 / static_cast<double>(MBIG);
    static int iff = 0;
    static int inext = 0;
    static int inextp = 0;
    static std::vector<int> ma(55);

    if (idum < 0 || iff == 0) {
        iff = 1;
        int mj = MSEED - std::abs(idum);
        mj %= MBIG;
        ma[54] = mj;
        int mk = 1;
        for (int i = 0; i < 54; i++) {
            int ii = (21 * (i + 1)) % 55;
            ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < MZ) mk += MBIG;
            mj = ma[ii - 1];
        }
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < 55; i++) {
                ma[i] -= ma[(i + 31) % 55];
                if (ma[i] < MZ) ma[i] += MBIG;
            }
        }
        inext = 0;
        inextp = 31;
        idum = 1;
    }
    inext = (inext + 1) % 55;
    inextp = (inextp + 1) % 55;
    int mj = ma[inext] - ma[inextp];
    if (mj < MZ) mj += MBIG;
    ma[inext] = mj;
    return static_cast<double>(mj) * FAC;
}

void shellsort(int n, int ind, int dim1, std::vector<int>& arr) {
    std::vector<double> varr(dim1); // Temporary array
    int inc = 1;
    while (inc <= n / 3) {
        inc = 3 * inc + 1;
    }
    while (inc > 0) {
        for (int i = inc; i < n; i++) {
            double v = arr[ind * dim1 + i];
            for (int j = 0; j < dim1; j++) {
                varr[j] = arr[j * dim1 + i];
            }
            int j = i;
            while (j >= inc && arr[ind * dim1 + (j - inc)] > v) {
                for (int k = 0; k < dim1; k++) {
                    arr[k * dim1 + j] = arr[k * dim1 + (j - inc)];
                }
                j -= inc;
            }
            for (int k = 0; k < dim1; k++) {
                arr[k * dim1 + j] = varr[k];
            }
        }
        inc /= 3;
    }
}

void piksrt(int n, std::vector<double>& arr) {
    for (int j = 1; j < n; j++) {
        double a = arr[j];
        int i = j - 1;
        while (i >= 0 && arr[i] > a) {
            arr[i + 1] = arr[i];
            i--;
        }
        arr[i + 1] = a;
    }
}
