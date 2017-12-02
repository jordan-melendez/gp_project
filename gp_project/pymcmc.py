import pymc3 as pm
import pandas as pd
import numpy as np

A_data = pd.read_csv("./data/A_data.csv")
A_data = A_data[A_data.Energy == 100]
theory_points = A_data.loc[:, ['2', '3', '4', '5']].values
kinpars = A_data.theta.values
kinpars = (kinpars - kinpars.min()) / (kinpars.max() - kinpars.min())
jitter = 1e-6
length_scales = 0.9 # TOTALLY MADE UP

model = pm.Model()

with model:
    sigmasq = pm.InverseGamma("sigmasq", alpha=1, beta=1)
    mu = pm.Normal("mu", mu=0, sd=1e3)
    Q = 0.36 #pm.Beta("Q", alpha=1, beta=1)
    rho = pm.Beta("rho", alpha=1, beta=1)

    R = rho ** (np.subtract.outer(kinpars, kinpars)**2)
    R += 1e-5 * np.diag(kinpars.shape)

    cov = sigmasq * R

    C2 = pm.MvNormal("c2", mu=mu * Q**2, cov=cov * Q**4, observed=theory_points[:, 0])
    # C3 = pm.MvNormal("c3", observed=theory_points[:, 1] / Q**3, mu=mu, cov= 1/sigmasq * R)
    # C4 = pm.MvNormal("c4", observed=theory_points[:, 2] / Q**4, mu=mu, cov= 1/sigmasq * R)
    # C5 = pm.MvNormal("c5", observed=theory_points[:, 3] / Q**5, mu=mu, cov= 1/sigmasq * R)

    trace = pm.sample()

    print(trace['Q'][-5:])

