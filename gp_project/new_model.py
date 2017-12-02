import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import cholesky
from functools import partial
from kronecker import gaussian_kernel
from hamMCMC import HamiltonianSampler
import pdb

chol_solve = partial(sp.linalg.cho_solve, check_finite=False)

index = {
    "Q": 0,
    "MU": 1,
    "SIGMASQ": 2
    # "DELTA": 3
}


def UFactory(Xs, kinpars, ls, jitter, mu_mu, sigmasq_mu, alpha_sig,
             beta_sig, alpha_Q, beta_Q):

    N, K = Xs.shape
    R = gaussian_kernel(kinpars, kinpars, ls) + jitter * np.eye(N)
    Rchol = np.linalg.cholesky(R)

    def logCi(position, i):
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Q = position[index["Q"]]
        # Since index starts at 0, we need to add 2 to get the correct order
        Ci = Xs[:, i] / Q**(i + 2)

        V = Ci - mu
        RinvV = chol_solve((Rchol, True), V)

        return N/2 * np.log(sigmasq) - 1/2 * V.T @ RinvV
        # return multivariate_normal.logpdf(Ci, mean=mu, cov=sigmasq*R)

    def logC(position):
        return sum(logCi(position, i) for i in range(K))

    def loglike(position):
        Q = position[index["Q"]]
        k = K + 2
        return logC(position) #- k*(k+1)/2 * np.log(Q)

    def logQ(position):
        Q = position[index["Q"]]
        return (alpha_Q - 1) * np.log(Q) + (beta_Q - 1) * np.log(1 - Q)
        # return sp.stats.beta.logpdf(Q, a=alpha_Q, b=beta_Q)

    def logMu(position):
        mu = position[index["MU"]]
        return -1/(2 * sigmasq_mu) * (mu - mu_mu)**2
        # return sp.stats.norm.logpdf(mu, loc=mu_mu, scale=np.sqrt(sigmasq_mu))

    def logSigmasq(position):
        sigmasq = position[index["SIGMASQ"]]
        return (alpha_sig - 1) * np.log(sigmasq) - beta_sig * sigmasq
        # return sp.stats.invgamma.logpdf(sigmasq, a=alpha_sig, scale=beta_sig)

    def U(position):
        return loglike(position) + \
               logQ(position) + \
               logMu(position) + \
               logSigmasq(position)

    return U


def GradUFactory(Xs, kinpars, ls, jitter, mu_mu, sigmasq_mu, alpha_sig,
                 beta_sig, alpha_Q, beta_Q):

    N, K = Xs.shape
    R = gaussian_kernel(kinpars, kinpars, ls) + jitter * np.eye(N)
    Rchol = np.linalg.cholesky(R)

    def dJacobiandQ(position):
        Q = position[index["Q"]]
        k = K + 2
        return -k * (k+1) / (2*Q)

    def dCidSigmasq(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Ci = Xs[:, i] / Q**(i+2)

        V = Ci - mu
        RinvV = chol_solve((Rchol, True), V)

        return N/2 * 1/sigmasq - 1/2 * V.T @ RinvV

    def dCidMu(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Ci = Xs[:, i] / Q**(i+2)

        V = Ci - mu
        RinvV = chol_solve((Rchol, True), V)

        return sigmasq * RinvV @ np.ones(N)

    def dCidQ(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)

        j = i+2
        Ci = Xs[:, i] / Q**j

        RinvV = chol_solve((Rchol, True), Ci)
        # pdb.set_trace()
        return -sigmasq/2 * (Ci - mu).T @ RinvV * -2*j/(Q**(j+1))

    def dCdSigmasq(position):
        return sum(dCidSigmasq(position, i) for i in range(K))

    def dCdMu(position):
        return sum(dCidMu(position, i) for i in range(K))

    def dCdQ(position):
        return sum(dCidQ(position, i) for i in range(K))

    def dlogQdQ(position):
        Q = position[index["Q"]]
        return (alpha_Q - 1) / Q - (beta_Q - 1) / (1 - Q)

    def dlogMudMu(position):
        mu = position[index["MU"]]
        return - 1/(2*sigmasq_mu) * (mu - mu_mu)

    def dlogSigmasqdSigmaSq(position):
        sigmasq = position[index["SIGMASQ"]]
        return (alpha_sig - 1) / sigmasq + beta_sig

    def dJointdQ(position):
        # return dDeltadQ(position) + dCdQ(position) + dlogQdQ(position)
        return dCdQ(position) + dlogQdQ(position)# + dJacobiandQ(position)

    def dJointdMu(position):
        # return dDeltadMu(position) + dCdMu(position) + dlogMudMu(position)
        return dCdMu(position) + dlogMudMu(position)

    def dJointdSigmasq(position):
        # return dDeltadSigmasq(position) + dCdSigmasq(position) + \
        #     dlogSigmasqdSigmaSq(position)
        return dCdSigmasq(position) + dlogSigmasqdSigmaSq(position)

    def gradU(position):
        # pdb.set_trace()
        return np.array([dJointdQ(position), dJointdMu(position),
                         dJointdSigmasq(position)])

    return gradU

A_data = pd.read_csv("./data/A_data.csv")
A_data = A_data[A_data.Energy == 100]
theory_points = A_data.loc[:, ['2', '3', '4', '5']].values
kinpars = A_data.theta.values
jitter = 1e-10
length_scales = 0.3  # TOTALLY MADE UP

MEAN_PRIOR_MEAN = 0
MEAN_PRIOR_VAR = 1e6

VAR_PRIOR_ALPHA = 1e-3
VAR_PRIOR_BETA = 1e-3

Q_PRIOR_ALPHA = 1
Q_PRIOR_BETA = 1

U_function = UFactory(
    theory_points, kinpars, length_scales, jitter,
    mu_mu=MEAN_PRIOR_MEAN, sigmasq_mu=MEAN_PRIOR_VAR,
    alpha_Q=Q_PRIOR_ALPHA, beta_Q=Q_PRIOR_BETA,
    alpha_sig=VAR_PRIOR_ALPHA, beta_sig=VAR_PRIOR_BETA)
grad_U_function = GradUFactory(
    theory_points, kinpars, length_scales, jitter,
    mu_mu=MEAN_PRIOR_MEAN, sigmasq_mu=MEAN_PRIOR_VAR,
    alpha_Q=Q_PRIOR_ALPHA, beta_Q=Q_PRIOR_BETA,
    alpha_sig=VAR_PRIOR_ALPHA, beta_sig=VAR_PRIOR_BETA)

SCALES = np.square(np.array([4e-5, 2e-3, 2e-4]))

bayes_model = HamiltonianSampler(U_function, grad_U_function, num_leaps=25, step_size=SCALES, tempering=np.array([1.01, 1.01, 1.01]))

start_position = np.array([0.3, 0, 0.01])  # Q, Mu, SigamSq

bayes_model.initialize(start_position)
bayes_model.set_bounds({0: (0, 1),
                        1: (-np.Infinity, np.Infinity),
                        2: (0, np.Infinity)})
bayes_model.set_seed(2343)
#bayes_model.burn_in(100)
results = bayes_model.sample(1000)

print(results)

Q = results[:, index["Q"]]
mu = results[:, index["MU"]]
sigmasq = results[:, index["SIGMASQ"]]

delta_mu = Q**6 / (1 - Q) * mu
delta_sig = np.sqrt(Q**12 / (1 - Q**2) * sigmasq)
deltas = np.random.normal(loc=delta_mu, scale=delta_sig)

matplotlib.rcParams['axes.formatter.useoffset'] = False

# ham = np.zeros([results.shape[0], 2])
# for i in range(ham.shape[0]):
#     ham[i,:] = bayes_model.evaluate_energy(results[i,:])

plt.plot(results[:, 3], results[:, 4])
plt.show()

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(results[:, 1], results[:, 2], results[:, 0])
plt.show()

plt.plot(Q)
plt.plot(mu)
plt.plot(sigmasq)
plt.show()

plt.plot(Q)
plt.show()
plt.plot(mu)
plt.show()
plt.plot(sigmasq)
plt.show()

# print(results) #print_res)

print("Acceptance Percentage: ", bayes_model.number_accepted / bayes_model.total_number_of_draws)