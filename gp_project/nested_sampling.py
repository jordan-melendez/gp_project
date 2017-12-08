import pymultinest
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymultinest.solve import solve
import scipy as sp
import scipy.stats
import os
from gp_project.kronecker import gaussian_kernel

if not os.path.exists("chains"):
    os.mkdir("chains")


def prior(cube):
    Q, mu, sigmasq = cube[0], cube[1], cube[2]
    # Hyperparameters
    Q_a = 1
    Q_b = 1
    mu_mu = 0
    mu_sigma = 1
    sigmasq_a = 1
    # Take interval 0:1 -> prior disribution through inverse CDF
    cube[0] = sp.stats.beta.ppf(Q, a=Q_a, b=Q_b)
    cube[1] = sp.stats.norm.ppf(mu, loc=mu_mu, scale=mu_sigma)
    cube[2] = sp.stats.invgamma.ppf(sigmasq, a=sigmasq_a)
    return cube


def loglike(cube):
    Q, mu, sigmasq = cube[0], cube[1], cube[2]
    mu = mu*np.ones(N)
    loglikes = []
    k = len(ydata) + 2
    for i, cn in enumerate(ydata):
        n = i + 2
        logpdf_n = sp.stats.multivariate_normal.logpdf(
            cn/Q**n, mean=mu, cov=sigmasq*R + jitter * np.eye(N))
        loglikes.append(logpdf_n)
    val = np.sum(loglikes) - k*(k+1)/2 * np.log(Q)
    return val

datafile = 'A_data'
df = pd.read_csv("./data/{}.csv".format(datafile), index_col=[0, 1])
idx = pd.IndexSlice
df = df.loc[100]
x = np.arange(10, 180, 10)
N = len(x)
ls = 35
jitter = 1e-6
R = gaussian_kernel(x, x, ls)
ydata = df.loc[x, '2':'5'].values.T


# number of dimensions our problem has
parameters = ["Q", "mu", "sigma"]
n_params = len(parameters)
# name of the output files
prefix = "chains/3-"
datafile = prefix + datafile + "_"

# run MultiNest
result = solve(
    LogLikelihood=loglike, Prior=prior,
    n_dims=n_params, outputfiles_basename=datafile, resume=False, verbose=True)
# json.dump(parameters, open(datafile + '_1_params.json', 'w'))  # save parameter names

# # plot the distribution of a posteriori possible models
# plt.figure()
# plt.plot(x, ydata.T, '+ ', color='red', label='data')
# a = pymultinest.Analyzer(outputfiles_basename=datafile, n_params=n_params)
# colors = ['orange', 'green', 'blue', 'red']
# for (Q, mu, sigmasq) in a.get_equal_weighted_posterior()[::100,:-1]:
#     for i, cn in enumerate(ydata):
#         n = i + 2
#         plt.plot(x, cn/Q**n, '-', color=colors[i],
#                  alpha=0.2, label='c{}'.format(n))

# # plt.legend()
# plt.tight_layout()
# plt.savefig(datafile + 'posterior.pdf')
# plt.close()
