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
    mu2, sigma2 = cube[0], cube[1]
    mu3, sigma3 = cube[2], cube[3]
    mu4, sigma4 = cube[4], cube[5]
    mu5, sigma5 = cube[6], cube[7]
    # Hyperparameters
    mu_mu = 0
    mu_sigma = 1
    sigmasq_a = 1
    sigma_s = 0.5
    sigma_mu = 0
    # Take interval 0:1 -> prior disribution through inverse CDF
    cube[0] = sp.stats.norm.ppf(mu2, loc=mu_mu, scale=mu_sigma)
    cube[1] = sp.stats.lognorm.ppf(sigma2, s=sigma_s, loc=sigma_mu)
    cube[2] = sp.stats.norm.ppf(mu3, loc=mu_mu, scale=mu_sigma)
    cube[3] = sp.stats.lognorm.ppf(sigma3, s=sigma_s, loc=sigma_mu)
    cube[4] = sp.stats.norm.ppf(mu4, loc=mu_mu, scale=mu_sigma)
    cube[5] = sp.stats.lognorm.ppf(sigma4, s=sigma_s, loc=sigma_mu)
    cube[6] = sp.stats.norm.ppf(mu5, loc=mu_mu, scale=mu_sigma)
    cube[7] = sp.stats.lognorm.ppf(sigma5, s=sigma_s, loc=sigma_mu)
    return cube


def loglike(cube):
    mu2, sigma2 = cube[0], cube[1]
    mu3, sigma3 = cube[2], cube[3]
    mu4, sigma4 = cube[4], cube[5]
    mu5, sigma5 = cube[6], cube[7]
    mu = [mu2, mu3, mu4, mu5]
    sigma = [sigma2, sigma3, sigma4, sigma5]
    loglikes = []
    k = len(ydata) + 2
    for i, dX in enumerate(ydata):
        n = i + 2
        logpdf_n = sp.stats.multivariate_normal.logpdf(
            dX/Q**n, mean=mu[i]*np.ones(N),
            cov=sigma[i]**2*R + jitter * np.eye(N))
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
ls = 35
jitter = 1e-6
R = gaussian_kernel(x, x, ls)
ydata = df.loc[x, '2':'5'].values.T
# print(ydata)


# number of dimensions our problem has
parameters = ["mu2", "sigma2", "mu3", "sigma3",
              "mu4", "sigma4", "mu5", "sigma5"]
n_params = len(parameters)
# name of the output files
prefix = "chains/not_iid-"
datafile = prefix + datafile + "_"

# run MultiNest
result = solve(
    LogLikelihood=loglike, Prior=prior,
    n_dims=n_params, outputfiles_basename=datafile, resume=False, verbose=True)
# json.dump(parameters, open(datafile + 'params.json', 'w'))  # save parameter names

# # plot the distribution of a posteriori possible models
# fig = plt.figure()
# cax = fig.add_subplot(131)
# # plt.plot(x, ydata.T, '+ ', color='red', label='data')
# a = pymultinest.Analyzer(outputfiles_basename=datafile, n_params=n_params)
# colors = ['orange', 'green', 'blue', 'red']
# mulist, siglist = zip(*a.get_equal_weighted_posterior()[::, :-1])
# for (mu, sigma) in a.get_equal_weighted_posterior()[::100, :-1]:
#     for i, dX in enumerate(ydata):
#         n = i + 2
#         plt.plot(x, dX/Q**n, '-', color=colors[i],
#                  alpha=0.2, label='c{}'.format(n))

# sig_ax = fig.add_subplot(132)
# mu_ax = fig.add_subplot(133)
# sig_ax.hist(siglist, bins='auto', normed=True, alpha=0.5)
# mu_ax.hist(mulist, bins='auto', normed=True, alpha=0.5)

# # plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig(datafile + 'posterior.pdf')
# plt.close()
