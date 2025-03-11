import numpy as np
from scipy.sparse.linalg import eigsh
import numba as nb
import os
import argparse

# Global variables
class LanczosInfo:
    niter = 0
    nkeep = 0
    lvec_file = None
    coef_file = 22
    reorthog = True
    writetodisk = True
    lvec = None

class Timing:
    timeorthog = 0.0
    timelast_ort = 0.0

# Supporting functions
@nb.jit(nopython=True)
def pythag(a, b):
    p = max(abs(a), abs(b))
    if p == 0.0:
        return 0.0
    r = (min(abs(a), abs(b)) / p) ** 2
    while True:
        t = 4.0 + r
        if t == 4.0:
            break
        s = r / t
        u = 1.0 + 2.0 * s
        p *= u
        r = (s / u) ** 2 * r
    return p

@nb.jit(nopython=True)
def tqli(nm, n, d, e, z):
    ierr = 0
    if n == 1:
        return ierr
    e[:-1] = e[1:]
    e[-1] = 0.0
    f = 0.0
    tst1 = 0.0
    for l in range(n):
        j = 0
        h = abs(d[l]) + abs(e[l])
        if tst1 < h:
            tst1 = h
        m = l
        while m < n:
            tst2 = tst1 + abs(e[m])
            if tst2 == tst1:
                break
            m += 1
        if m == l:
            d[l] += f
            continue
        while True:
            if j == 30:
                ierr = l + 1
                return ierr
            j += 1
            l1 = l + 1
            l2 = l1 + 1
            g = d[l]
            p = (d[l1] - g) / (2.0 * e[l])
            r = pythag(p, 1.0)
            d[l] = e[l] / (p + np.copysign(r, p))
            d[l1] = e[l] * (p + np.copysign(r, p))
            dl1 = d[l1]
            h = g - d[l]
            if l2 < n:
                d[l2:] -= h
            f += h
            p = d[m]
            c = 1.0
            c2 = c
            el1 = e[l1]
            s = 0.0
            mml = m - l
            for ii in range(mml):
                c3 = c2
                c2 = c
                s2 = s
                i = m - 1 - ii
                g = c * e[i]
                h = c * p
                r = pythag(p, e[i])
                e[i + 1] = s * r
                s = e[i] / r
                c = p / r
                p = c * d[i] - s * g
                d[i + 1] = h + s * (c * g + s * d[i])
                for k in range(nm):
                    h = z[k, i + 1]
                    z[k, i + 1] = s * z[k, i] + c * h
                    z[k, i] = c * z[k, i] - s * h
            p = -s * s2 * c3 * el1 * e[l] / dl1
            e[l] = s * p
            d[l] = c * p
            tst2 = tst1 + abs(e[l])
            if tst2 <= tst1:
                break
        d[l] += f
    indices = np.argsort(d)
    d[:] = d[indices]
    z[:, :] = z[:, indices]
    return ierr

@nb.jit(nopython=True)
def eigsrt(d, v, n, niter_max):
    indices = np.argsort(d[:n])[::-1]  # Sort only first n elements, descending
    d[:n] = d[:n][indices]             # Update first n elements of d
    v[:, :n] = v[:, indices]           # Update first n columns of v

# File I/O functions (serial)
def open_file():
    LanczosInfo.lvec_file = open("lanczosvector.lvec", "wb+")

def close_file():
    if LanczosInfo.lvec_file:
        LanczosInfo.lvec_file.seek(0)
        np.array([0], dtype=np.int32).tofile(LanczosInfo.lvec_file)
        LanczosInfo.lvec_file.close()
        LanczosInfo.lvec_file = None

def write_file(v, dimbasis, i):
    if i == 1:
        LanczosInfo.lvec_file.seek(0)
    v.tofile(LanczosInfo.lvec_file)

def read_file(temp_v, dimbasis, i):
    if i == 1:
        LanczosInfo.lvec_file.seek(0)
    temp_v[:] = np.fromfile(LanczosInfo.lvec_file, dtype=np.float64, count=dimbasis)

# Parallelized subroutines
@nb.jit(nopython=True, parallel=True)
def applyh(n, h, vecin, vecout):
    for j in nb.prange(n):
        vecout[j] = 0.0
        for i in range(n):
            vecout[j] += h[i, j] * vecin[i]

@nb.jit(nopython=True, parallel=True)
def dnormvec(dvec):
    dnorm = 0.0
    for i in nb.prange(len(dvec)):
        d = dvec[i]
        dnorm += d * d
    dnorm = np.sqrt(dnorm)
    d = 1.0 / dnorm
    for i in nb.prange(len(dvec)):
        dvec[i] *= d
    return dnorm

@nb.jit(nopython=True, parallel=True)
def dvecproj(dvec1, dvec2):
    dsclrprod = 0.0
    for i in nb.prange(len(dvec1)):
        d1 = dvec1[i]
        d2 = dvec2[i]
        dsclrprod += d1 * d2
    for i in nb.prange(len(dvec1)):
        dvec1[i] -= dvec2[i] * dsclrprod
    return dsclrprod

@nb.jit(nopython=True, parallel=True)
def reorthogonalize_core(w, v, n, previous_vectors):
    for i in nb.prange(n):
        w[i] = w[i]
    for j in range(len(previous_vectors)):
        temp_v = previous_vectors[j]
        dsclrprod = 0.0
        for i in nb.prange(n):
            dsclrprod += w[i] * temp_v[i]
        for i in nb.prange(n):
            w[i] -= temp_v[i] * dsclrprod
    for i in nb.prange(n):
        v[i] = w[i]

def reorthogonalize(w, v, n, iter):
    previous_vectors = []
    for j in range(1, iter + 1):
        temp_v = np.zeros(n)
        read_file(temp_v, n, j)
        previous_vectors.append(temp_v)
    reorthogonalize_core(w, v, n, np.array(previous_vectors))

# Main Lanczos algorithm
def main(num_threads=None):
    if num_threads is not None:
        nb.set_num_threads(num_threads)
    print(f"Using {nb.get_num_threads()} Numba threads")

    LanczosInfo.writetodisk = True
    LanczosInfo.reorthog = True

    if LanczosInfo.writetodisk:
        open_file()

    n = 10000  # Reduce to 1000 for testing
    LanczosInfo.nkeep = 5
    LanczosInfo.niter = 50

    if not LanczosInfo.writetodisk:
        LanczosInfo.lvec = np.zeros((n, LanczosInfo.niter))
    h = np.zeros((n, n))
    v = np.zeros(n)
    w = np.zeros(n)
    alpha = np.zeros(LanczosInfo.niter)
    beta = np.zeros(LanczosInfo.niter)
    d = np.zeros(LanczosInfo.niter)
    e = np.zeros(LanczosInfo.niter)
    z = np.zeros((n, LanczosInfo.niter))

    # Optimized random symmetric matrix generation
    upper_tri_size = (n * (n + 1)) // 2  # Number of elements in upper triangle + diagonal
    upper_tri = np.random.rand(upper_tri_size)  # Single call to generate all values
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            h[i, j] = upper_tri[idx]
            if i != j:  # Avoid duplicating diagonal
                h[j, i] = h[i, j]
            idx += 1
    #for j in range(n):
    #    for i in range(j + 1):
    #        h[i, j] = np.random.rand()
    #        h[j, i] = h[i, j]

    dnorm = 1.0 / np.sqrt(float(n))
    v[:] = dnorm
    da = dnormvec(v)

    iter = 0
    while iter < LanczosInfo.niter:
        iter += 1
        write_file(v, n, iter)
        applyh(n, h, v, w)
        da = dvecproj(w, v)
        alpha[iter - 1] = da

        if LanczosInfo.reorthog:
            reorthogonalize(w, v, n, iter)

        if iter < LanczosInfo.niter:
            db = dnormvec(w)
            beta[iter] = db
            v[:] = w

        d.fill(0.0)
        e.fill(0.0)
        z.fill(0.0)
        for j in range(LanczosInfo.niter):
            if j < LanczosInfo.niter:
                z[j, j] = 1.0
        d[:iter] = alpha[:iter]
        e[:iter] = beta[:iter]

        ierr = tqli(n, iter, d, e, z)
        eigsrt(d, z, iter, LanczosInfo.niter)

        if ierr != 0:
            print(f"diag ierr={ierr}")
        print(f"{LanczosInfo.nkeep} lowest eigenvalues - Lanczos")
        print(f"iteration: {iter}")
        #for i in range(min(LanczosInfo.nkeep, iter)):
        for i in range(LanczosInfo.nkeep):
            print(f"{i + 1} {d[i]}")

    print("Lanczos iterations finished...")
    if LanczosInfo.writetodisk:
        close_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanczos with Numba parallelism")
    parser.add_argument("--threads", type=int, default=None, help="Number of Numba threads")
    args = parser.parse_args()
    main(num_threads=args.threads)

