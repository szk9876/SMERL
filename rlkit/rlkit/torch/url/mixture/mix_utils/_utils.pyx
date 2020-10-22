import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def grouped_mean(ndarray[np.float32_t, ndim=2] dists not None, ndarray[np.int_t, ndim=1] group not None):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = dists.shape[0]
    cdef Py_ssize_t d = dists.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] means = np.zeros((group.max()+1, d), dtype=np.float32)
    cdef np.ndarray[np.int16_t, ndim=1] counts = np.zeros((group.max()+1), dtype=np.int16)

    
    for i in range(n):
        g = group[i]
        means[g] += dists[i]
        counts[g] += 1

    for i in range(n):
        g = group[i]
        dists[i] = means[g] / counts[g]

    return dists


@cython.boundscheck(False)
def grouped_mean_part(
    ndarray[np.float32_t, ndim=2] dists not None,
    ndarray[np.int_t, ndim=1] group not None,
    int m1,
    int m2):

    cdef Py_ssize_t i
    cdef Py_ssize_t n = dists.shape[0]
    cdef Py_ssize_t d = dists.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] means = np.zeros((group.max()+1, d), dtype=np.float32)
    cdef np.ndarray[np.int16_t, ndim=1] counts = np.zeros((group.max()+1), dtype=np.int16)

    
    for i in range(n):
        g = group[i]
        if g >= m1 and g < m2:
            means[g] += dists[i]
            counts[g] += 1

    for i in range(n):
        g = group[i]
        if g >= m1 and g < m2:
            dists[i] = means[g] / counts[g]

    return dists
