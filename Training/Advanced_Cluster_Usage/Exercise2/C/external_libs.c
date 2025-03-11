#include <math.h>
#include <stdlib.h>     
#include <stdio.h>

void tred2(int nm, int n, double *a, double *d, double *e, double *z) {
    int i, j, k, l, ii, jp1;
    double f, g, h, hh, scale;

    // Copy lower triangle of A to Z and initialize D
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
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
    for (ii = 1; ii < n; ii++) {
        i = n - 1 - (ii - 1); // Adjusted for 0-based indexing
        l = i - 1;
        h = 0.0;
        scale = 0.0;

        // Compute scale factor
        if (l >= 0) { // Adjusted for 0-based indexing
            for (k = 0; k <= l; k++) {
                scale += fabs(d[k]);
            }
        }

        if (scale == 0.0) {
            e[i] = d[l];
            for (j = 0; j <= l; j++) {
                d[j] = z[l * nm + j];
                z[i * nm + j] = 0.0;
                z[j * nm + i] = 0.0;
            }
        } else {
            // Scale the vector and compute h
            for (k = 0; k <= l; k++) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            f = d[l];
            g = -copysign(sqrt(h), f);
            e[i] = scale * g;
            h -= f * g;
            d[l] = f - g;

            // Form A*U
            for (j = 0; j <= l; j++) {
                e[j] = 0.0;
            }
            for (j = 0; j <= l; j++) {
                f = d[j];
                z[j * nm + i] = f;
                g = e[j] + z[j * nm + j] * f;
                jp1 = j + 1;
                if (jp1 <= l) {
                    for (k = jp1; k <= l; k++) {
                        g += z[k * nm + j] * d[k];
                        e[k] += z[k * nm + j] * f;
                    }
                }
                e[j] = g;
            }

            // Form P and Q
            f = 0.0;
            for (j = 0; j <= l; j++) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            hh = f / (h + h);
            for (j = 0; j <= l; j++) {
                e[j] -= hh * d[j];
            }

            // Reduce A
            for (j = 0; j <= l; j++) {
                f = d[j];
                g = e[j];
                for (k = j; k <= l; k++) {
                    z[k * nm + j] -= f * e[k] + g * d[k];
                }
                d[j] = z[l * nm + j];
                z[i * nm + j] = 0.0;
            }
        }
        d[i] = h;
    }

    // Accumulate transformation matrices
    for (i = 1; i < n; i++) {
        l = i - 1;
        z[(n - 1) * nm + l] = z[l * nm + l];
        z[l * nm + l] = 1.0;
        h = d[i];
        if (h != 0.0) {
            for (k = 0; k <= l; k++) {
                d[k] = z[k * nm + i] / h;
            }
            for (j = 0; j <= l; j++) {
                g = 0.0;
                for (k = 0; k <= l; k++) {
                    g += z[k * nm + i] * z[k * nm + j];
                }
                for (k = 0; k <= l; k++) {
                    z[k * nm + j] -= g * d[k];
                }
            }
        }
        for (k = 0; k <= l; k++) {
            z[k * nm + i] = 0.0;
        }
    }

    // Finalize output
    for (i = 0; i < n; i++) {
        d[i] = z[(n - 1) * nm + i];
        z[(n - 1) * nm + i] = 0.0;
    }
    z[(n - 1) * nm + (n - 1)] = 1.0;
    e[0] = 0.0;
}

double pythag(double a, double b); // Forward declaration

void tqli(int nm, int n, double *d, double *e, double *z, int *ierr) {
    int i, j, k, l, m, ii, l1, l2, mml;
    double c, c2, c3, dl1, el1, f, g, h, p, r, s, s2, tst1, tst2;

    *ierr = 0;
    if (n == 1) return;

    // Shift subdiagonal elements
    for (i = 1; i < n; i++) {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    f = 0.0;
    tst1 = 0.0;

    // Main QL iteration
    for (l = 0; l < n; l++) {
        j = 0;
        h = fabs(d[l]) + fabs(e[l]);
        if (tst1 < h) tst1 = h;
        m = l;
        while (m < n) {
            tst2 = tst1 + fabs(e[m]);
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
            l1 = l + 1;
            l2 = l1 + 1;
            g = d[l];
            p = (d[l1] - g) / (2.0 * e[l]);
            r = pythag(p, 1.0);
            d[l] = e[l] / (p + copysign(r, p));
            d[l1] = e[l] * (p + copysign(r, p));
            dl1 = d[l1];
            h = g - d[l];
            if (l2 < n) {
                for (i = l2; i < n; i++) {
                    d[i] -= h;
                }
            }
            f += h;

            // QL transformation
            p = d[m];
            c = 1.0;
            c2 = c;
            el1 = e[l1];
            s = 0.0;
            mml = m - l;
            for (ii = 0; ii < mml; ii++) {
                c3 = c2;
                c2 = c;
                s2 = s;
                i = m - 1 - ii;
                g = c * e[i];
                h = c * p;
                r = pythag(p, e[i]);
                e[i + 1] = s * r;
                s = e[i] / r;
                c = p / r;
                p = c * d[i] - s * g;
                d[i + 1] = h + s * (c * g + s * d[i]);
                for (k = 0; k < n; k++) {
                    h = z[(i + 1) * nm + k];
                    z[(i + 1) * nm + k] = s * z[i * nm + k] + c * h;
                    z[i * nm + k] = c * z[i * nm + k] - s * h;
                }
            }
            p = -s * s2 * c3 * el1 * e[l] / dl1;
            e[l] = s * p;
            d[l] = c * p;
            tst2 = tst1 + fabs(e[l]);
            if (tst2 <= tst1) break;
        } while (1);
        d[l] += f;
    }

    // Order eigenvalues and eigenvectors
    for (ii = 1; ii < n; ii++) {
        i = ii - 1;
        k = i;
        p = d[i];
        for (j = ii; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (j = 0; j < n; j++) {
                p = z[i * nm + j];
                z[i * nm + j] = z[k * nm + j];
                z[k * nm + j] = p;
            }
        }
    }
}

double pythag(double a, double b) {
    double p, r, s, t, u;
    p = fmax(fabs(a), fabs(b));
    if (p == 0.0) {
        return 0.0;
    }
    r = pow(fmin(fabs(a), fabs(b)) / p, 2);
    while (1) {
        t = 4.0 + r;
        if (t == 4.0) break;
        s = r / t;
        u = 1.0 + 2.0 * s;
        p *= u;
        r = pow(s / u, 2) * r;
    }
    return p;
}

void eigsrt(double *d, double *v, int n, int np) {
    int i, j, k;
    double p;
    for (i = 0; i < n - 1; i++) {
        k = i;
        p = d[i];
        for (j = i + 1; j < n; j++) {
            if (d[j] >= p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (j = 0; j < n; j++) {
                p = v[i * np + j];
                v[i * np + j] = v[k * np + j];
                v[k * np + j] = p;
            }
        }
    }
}

double ran3(int *idum) {
    const int MBIG = 1000000000;
    const int MSEED = 161803398;
    const int MZ = 0;
    const double FAC = 1.0 / MBIG;
    static int iff = 0;
    static int inext, inextp;
    static int ma[55];
    int i, ii, k, mj, mk;

    if (*idum < 0 || iff == 0) {
        iff = 1;
        mj = MSEED - abs(*idum);
        mj %= MBIG;
        ma[54] = mj;
        mk = 1;
        for (i = 0; i < 54; i++) {
            ii = (21 * (i + 1)) % 55;
            ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < MZ) mk += MBIG;
            mj = ma[ii - 1];
        }
        for (k = 0; k < 4; k++) {
            for (i = 0; i < 55; i++) {
                ma[i] -= ma[(i + 31) % 55];
                if (ma[i] < MZ) ma[i] += MBIG;
            }
        }
        inext = 0;
        inextp = 31;
        *idum = 1;
    }
    inext = (inext + 1) % 55;
    inextp = (inextp + 1) % 55;
    mj = ma[inext] - ma[inextp];
    if (mj < MZ) mj += MBIG;
    ma[inext] = mj;
    return mj * FAC;
}

void shellsort(int n, int ind, int dim1, int *arr) {
    int i, j, inc;
    double v;
    double *varr = malloc(dim1 * sizeof(double)); // Temporary array

    inc = 1;
    while (inc <= n / 3) {
        inc = 3 * inc + 1;
    }
    while (inc > 0) {
        for (i = inc; i < n; i++) {
            v = arr[ind * dim1 + i];
            for (j = 0; j < dim1; j++) {
                varr[j] = arr[j * dim1 + i];
            }
            j = i;
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
    free(varr);
}

void piksrt(int n, double *arr) {
    int i, j;
    double a;
    for (j = 1; j < n; j++) {
        a = arr[j];
        i = j - 1;
        while (i >= 0 && arr[i] > a) {
            arr[i + 1] = arr[i];
            i--;
        }
        arr[i + 1] = a;
    }
}


