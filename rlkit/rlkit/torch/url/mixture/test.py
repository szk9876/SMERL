import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.stats
from scipy import linalg


import models
from models import GaussianMixture as GMM
from models import BayesianGaussianMixture as BGMM

import os
import sys

data = np.load('./data_itr20.pkl')

X = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
N, T = data.shape[:2]


import itertools
# color_iter = itertools.cycle(
#     ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
color_iter = itertools.cycle([
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#808080'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(
            zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.3)
        splot.add_artist(ell)

    plt.title(title)


def cluster(mix_type, group=None, C=20):
    mix = getattr(models, mix_type)(
        n_components=C, covariance_type='full', verbose=1).fit(X, group=group)

    plot_results(X, mix.predict(X, group=group), mix.means_, mix.covariances_, 0,
                 '%s C=%s T=%s' % (mix_type, C, group))
    plt.show()


C = 100
print(T)
# cluster('GaussianMixture', C=C)
group = np.arange(X.shape[0]) // T

cluster('GaussianMixture', C=C, group=group)
cluster('BayesianGaussianMixture', C=C, group=group)

# cluster('BayesianGaussianMixture', C=C)
# cluster('BayesianGaussianMixture', C=C, group=T)
