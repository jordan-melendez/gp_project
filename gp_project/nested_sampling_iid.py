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
    mu, sigma = cube[0], cube[1]
    # Hyperparameters
    mu_mu = 0
    mu_sigma = 1
    sigmasq_a = 1
    # Take interval 0:1 -> prior disribution through inverse CDF
    cube[0] = sp.stats.norm.ppf(mu, loc=mu_mu, scale=mu_sigma)
    # cube[1] = sp.stats.invgamma.ppf(sigmasq, a=sigmasq_a)
    cube[1] = sp.stats.lognorm.ppf(sigma, s=0.5, loc=0)
    return cube


def loglike(cube):
    mu, sigma = cube[0], cube[1]
    mu = mu*np.ones(N)
    loglikes = []
    k = len(ydata) + 2
    for i, dX in enumerate(ydata):
        n = i + 2
        logpdf_n = sp.stats.multivariate_normal.logpdf(
            dX/Q**n, mean=mu, cov=sigma**2*R + jitter * np.eye(N))
        loglikes.append(logpdf_n)
    val = np.sum(loglikes) - k*(k+1)/2 * np.log(Q)
    return val

datafile = 'A_data'
df = pd.read_csv("./data/{}.csv".format(datafile), index_col=[0, 1])
idx = pd.IndexSlice
df = df.loc[150]
x = np.arange(10, 180, 10)
N = len(x)
Q = df.loc[x, 'Q'].values
Q = Q[0]
print(Q)
ls = 20
jitter = 1e-6
R = gaussian_kernel(x, x, ls)
ydata = df.loc[x, '2':'5'].values.T
# print(ydata)


# number of dimensions our problem has
parameters = ["mu", "sigma"]
n_params = len(parameters)
# name of the output files
prefix = "chains/iid-"
datafile = prefix + datafile + "_"

# run MultiNest
result = solve(
    LogLikelihood=loglike, Prior=prior,
    n_dims=n_params, outputfiles_basename=datafile, resume=False, verbose=True)
json.dump(parameters, open(datafile + 'params.json', 'w'))  # save parameter names

# plot the distribution of a posteriori possible models
fig = plt.figure()
cax = fig.add_subplot(131)
# plt.plot(x, ydata.T, '+ ', color='red', label='data')
a = pymultinest.Analyzer(outputfiles_basename=datafile, n_params=n_params)
colors = ['orange', 'green', 'blue', 'red']
mulist, siglist = zip(*a.get_equal_weighted_posterior()[::, :-1])
for (mu, sigma) in a.get_equal_weighted_posterior()[::100, :-1]:
    for i, dX in enumerate(ydata):
        n = i + 2
        plt.plot(x, dX/Q**n, '-', color=colors[i],
                 alpha=0.2, label='c{}'.format(n))

sig_ax = fig.add_subplot(132)
mu_ax = fig.add_subplot(133)
sig_ax.hist(siglist, bins='auto', normed=True, alpha=0.5)
mu_ax.hist(mulist, bins='auto', normed=True, alpha=0.5)

# plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(datafile + 'posterior.pdf')
plt.close()
