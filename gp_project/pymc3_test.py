import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pymc3 as pm
import scipy as sp
import scipy.stats

datafile = 'A_data'
# datafile = 'diff_cross_section_data'
df = pd.read_csv("./data/{}.csv".format(datafile), index_col=[0, 1])
# idx = pd.IndexSlice
df = df.loc[150]
x = np.arange(10, 180, 10)
N = len(x)
jitter = 1e-5
k = 5

ydata = df.loc[x, '2':'5'].values.T

dk_X = np.arange(1, 180, 1)
X5 = df.loc[dk_X, '0':'5'].values
X5 = np.sum(X5, axis=1)
deltak_exp = (df.loc[dk_X, 'exp'].values - X5).ravel()
Q = df.loc[x, 'Q'].values
Q = Q[0]
print(Q)
c2 = ydata[0]/Q**2
c3 = ydata[1]/Q**3
c4 = ydata[2]/Q**4
c5 = ydata[3]/Q**5
c = np.array([c2, c3, c4, c5])
# c = c2
# print(c)

# sd_hyperparams = {'mu': 0, 'sd': 0.5}
# ls_hyperparams = {'mu': np.log(25), 'sd': 0.5}
learn_Q = True

with pm.Model() as model:
    # Define priors
    # sdsq = pm.InverseGamma('sdsq', alpha=1, beta=1)
    sd = pm.Lognormal('sd', mu=0, sd=0.5)
    ls = pm.Lognormal('ls', mu=np.log(25), sd=0.5)
    # ls = 10
    mu = pm.Normal('mu', mu=0, sd=1)
    # mu = 0
    if learn_Q:
        Q = pm.Beta('Q', alpha=2, beta=2)
    scale = 0.4  # For numerics

    cov = sd**2 * pm.gp.cov.ExpQuad(1, ls=ls)
    if not learn_Q:
        # Set up model
        mean = pm.gp.mean.Constant(mu)
        gp = pm.gp.Marginal(mean_func=mean, cov_func=cov)
        gp.marginal_likelihood('cn', X=x[:, None], y=c, noise=jitter)

    if learn_Q:
        Qs = Q/scale
        gp2 = pm.gp.Marginal(
            mean_func=pm.gp.mean.Constant(Qs**2*mu), cov_func=Qs**4 * cov)
        gp3 = pm.gp.Marginal(
            mean_func=pm.gp.mean.Constant(Qs**3*mu), cov_func=Qs**6 * cov)
        gp4 = pm.gp.Marginal(
            mean_func=pm.gp.mean.Constant(Qs**4*mu), cov_func=Qs**8 * cov)
        gp5 = pm.gp.Marginal(
            mean_func=pm.gp.mean.Constant(Qs**5*mu), cov_func=Qs**10 * cov)
        gp = [gp2, gp3, gp4, gp5]

        gp2.marginal_likelihood(
            'c2', X=x[:, None], y=ydata[0]/scale**2, noise=jitter)
        gp3.marginal_likelihood(
            'c3', X=x[:, None], y=ydata[1]/scale**3, noise=jitter)
        gp4.marginal_likelihood(
            'c4', X=x[:, None], y=ydata[2]/scale**4, noise=jitter)
        gp5.marginal_likelihood(
            'c5', X=x[:, None], y=ydata[3]/scale**5, noise=jitter)

    trace = pm.sample(500, tune=1000, njobs=5)

delta_list = []
for t in trace:
    # print(t)
    delta_mu = t['mu']
    delta_sd = t['sd']
    if 'Q' in t:
        Q = t['Q']
    delta_mu *= Q**(k+1) / (1-Q)
    delta_sd *= Q**(k+1) / np.sqrt(1-Q**2)

    delta = sp.stats.norm.rvs(loc=delta_mu, scale=delta_sd, size=1)
    delta_list.append(delta[0])

# Save trace and relevant variables for later
with open('gp_model.pkl', 'wb') as buff:
    pickle.dump({'model': model,
                 'trace': trace,
                 'delta_list': delta_list,
                 'deltak_exp': deltak_exp,
                 'ydata': ydata,
                 'x': x,
                 'dk_X': dk_X,
                 'datafile': datafile,
                 'k': k,
                 'Q': Q,
                 'c': c,
                 'jitter': jitter,
                 'gp': gp,
                 'learn_Q': learn_Q,
                 'scale': scale
                 }, buff)
