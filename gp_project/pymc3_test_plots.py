import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pymc3 as pm
import scipy as sp
import scipy.stats

with open('gp_model.pkl', 'rb') as buff:
    data = pickle.load(buff)

model = data['model']
trace = data['trace']
delta_list = data['delta_list']
deltak_exp = data['deltak_exp']
ydata = data['ydata']
x = data['x']
dk_X = data['dk_X']
datafile = data['datafile']
k = data['k']
Q = data['Q']
c = data['c']
jitter = data['jitter']
gp = data['gp']
learn_Q = data['learn_Q']
# scale = data['scale']
scale = 0.4

Xnew = np.arange(1, 180, 5)


dfig = plt.figure(figsize=(4, 4))
dax = dfig.add_subplot(111)
# print(delta_list)
kde = sp.stats.gaussian_kde(delta_list)
delta_kde_domain = np.linspace(np.min(delta_list), np.max(delta_list), 1000)
dax.hist(delta_list, bins='auto', normed=True, alpha=0.5,
         label=r'pr($\Delta_k|\theta$)')
dax.plot(delta_kde_domain, kde(delta_kde_domain), color='blue')
dax.hist(deltak_exp, bins='auto', normed=True, alpha=0.5,
         label=r'Exp. $\Delta_k$')
dfig.canvas.draw()
dax.legend()
dax.set_xlabel(r'$\Delta_k$')
dax.set_title('Posterior Check')

nvars = len(model.free_RVs)
print(pm.summary(trace))
fig, ax = plt.subplots(nvars, 2, figsize=(8, 2*nvars))
if ax.ndim == 1:
    ax = np.array([ax])
pm.traceplot(trace, ax=ax,
             # priors=[pm.Lognormal.dist(**sd_hyperparams)]
             )

colors = ['orange', 'green', 'blue', 'red']
datafig = plt.figure(figsize=(4, 4))
dataax = datafig.add_subplot(111)
for i, cn in enumerate(c):
    n = i + 2
    dataax.plot(x, cn, marker='o', ls='None', color=colors[i],
                label=r'$c_{}$'.format({n}))
dataax.legend()
dataax.set_xlabel(r'$\theta$')
dataax.set_ylabel(r'$c_n$')

cfig = plt.figure(figsize=(4, 4))
cax = cfig.add_subplot(111)
cax.set_xlabel(r'$\theta$')
cax.set_ylabel(r'$c_n$')

gp_list = []
samples = 1
if not learn_Q:
    gp = [gp for i in c]
    samples = 50
    with model:
        for i, cn in enumerate(c):
            n = i + 2
            gp_list.append(
                gp[i].conditional(
                    "c{}new".format(n), Xnew=Xnew[:, None],
                    given={'y': cn, 'X': x[:, None], 'noise': jitter}
                    )
            )
        cn_samples = pm.sample_ppc(trace, vars=gp_list, samples=samples)
else:
    with model:
        for i, yn in enumerate(ydata):
            n = i + 2
            gp_list.append(
                gp[i].conditional(
                    "c{}new".format(n), Xnew=Xnew[:, None],
                    # given={'y': yn/scale**n, 'X': x[:, None], 'noise': jitter}
                    )
            )
        cn_samples = pm.sample_ppc(trace, vars=gp_list, samples=samples)
# if learn_Q:
#     for i, cn in enumerate(cn_samples):
#         n = i+2
#         cn = cn_samples['c{}new'.format(n)]

for i, _ in enumerate(cn_samples):
    n = i + 2
    cax.plot(x, c[i], marker='o', ls='None', color=colors[i],
             label=r'$c_{}$'.format({n}))
    cn = cn_samples['c{}new'.format(n)].T
    if learn_Q:
        cn_all = []
        for j, Q in enumerate(trace['Q']):
            if j % 5 == 0:
                Qs = Q/scale
                cn_all.append(Qs**n * cn)
        cn = np.hstack(cn_all)
    cax.plot(Xnew, cn, color=colors[i],
             alpha=0.1, lw=0.4)
cax.legend()
cax.set_ylim(1.2*np.min(c), 1.2*np.max(c))

# plt.show()
datafig.savefig('pymc3_{}_data_Q_{}.pdf'.format(datafile, learn_Q))
cfig.savefig('pymc3_{}_coeffs_Q_{}.pdf'.format(datafile, learn_Q))
fig.savefig("pymc3_{}_gp_model_Q_{}.pdf".format(datafile, learn_Q))
dfig.savefig("pymc3_{}_delta_gp_model_Q_{}.pdf".format(datafile, learn_Q))
