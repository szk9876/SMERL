import numpy as np
import numba
from numba import njit, prange

from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


@njit(nogil=True)
def _jit_grouped_mean(dists, m1, m2, group):
    for g in range(m1, m2):
        mask = group == g
        dists[mask] = dists[mask].sum(axis=0) / mask.sum()


def grouped_mean(dists, group):
    n_jobs = 5
    mm1, mm2 = group.min(), group.max()
    aa = np.arange(mm1, mm2 + 2, (mm2 - mm1 + 1) / n_jobs).astype(np.int)
    aa = [(aa[i], aa[i + 1]) for i in range(aa.shape[0] - 1)]

    # print(aa)
    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_jit_grouped_mean)(dists, m1, m2, group) for (m1, m2) \
            in aa)

    return dists

def const_grouped_mean(dists, group):
    # group = (group == 0).sum()
    dists_grouped = dists.reshape(dists.shape[0] // group, group,
                                  dists.shape[1])
    dists_grouped = dists_grouped.mean(1)[:, None].repeat(group, axis=1)
    dists_grouped = dists_grouped.reshape(
        dists_grouped.shape[0] * dists_grouped.shape[1],
        dists_grouped.shape[2])

    return dists_grouped

def check_group(group):
    aa = np.bincount(group)
    if np.all(aa[0] == aa):
        return aa[0]
    else:
        return group