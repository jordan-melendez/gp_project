import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from functools import partial
from kronecker import gaussian_kernel
from hamMCMC import HamiltonianSampler

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

    # def logDelta(position):
    #     sigsq = position[index["SIGMASQ"]]
    #     mu = position[index["MU"]]
    #     Q = position[index["Q"]]
    #     # delta = position[index['DELTA']]

    #     V = delta - mu / (1 - Q)
    #     RinvV = chol_solve((Rchol, True), V)

    #     return -N/2 * np.log(sigsq) \
    #            - N/2 * Q**(2*K + 2) / (1 - Q**2) \
    #            - 1/(2*sigsq) * V.T @ RinvV * (1-Q**2) / Q**(2*K + 2)

    def logCi(position, i):
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Q = position[index["Q"]]
        # Since index starts at 0, we need to add 2 to get the correct order
        Ci = Xs[:, i] / Q**(i + 2)

        V = Ci - mu
        RinvV = chol_solve((Rchol, True), V)

        return -N/2 * np.log(sigmasq) - 1/(2 * sigmasq) * V.T @ RinvV
        # return multivariate_normal.logpdf(Ci, mean=mu, cov=sigmasq*R)

    def logC(position):
        return sum(logCi(position, i) for i in range(K))

    def loglike(position):
        Q = position[index["Q"]]
        k = K + 2
        return logC(position) - k*(k+1)/2 * np.log(Q)

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
        return (-alpha_sig - 1) * np.log(sigmasq) - beta_sig * sigmasq
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

    # def dDeltadSigmasq(position):
    #     Q = position[index["Q"]]
    #     sigmasq = position[index["SIGMASQ"]]
    #     mu = position[index["MU"]] * np.ones(N)
    #     delta = position[index['DELTA']]

    #     V = delta - mu / (1-Q)
    #     RinvV = chol_solve((Rchol, True), V)

    #     return -N/2 * 1/sigmasq + 1/2 * 1/sigmasq**2 * V @ RinvV * \
    #         (1 - Q**2) / Q**(2*(K+2 + 1))

    # def dDeltadMu(position):
    #     Q = position[index["Q"]]
    #     sigmasq = position[index["SIGMASQ"]]
    #     mu = position[index["MU"]] * np.ones(N)
    #     delta = position[index['DELTA']]

    #     V = delta - mu / (1-Q)
    #     RinvV = chol_solve((Rchol, True), V)

    #     # TODO Double check the value K, K+2, etc.
    #     return 1/sigmasq * RinvV * 1/(1-Q)**2 * (1-Q**2)/Q**(2*(K+2 + 1))

    # def dDeltadQ(position):
    #     Q = position[index["Q"]]
    #     sigmasq = position[index["SIGMASQ"]]
    #     mu = position[index["MU"]] * np.ones(N)
    #     delta = position[index['DELTA']]

    #     k = K + 2

    #     V = delta - mu / (1-Q)
    #     RinvV = chol_solve((Rchol, True), V)

    #     first = (1-Q**2)/Q**(2*k+2) * \
    #         (2 * (k+1) * Q**(2*k + 1) * (1-Q**2) + 2*Q**(2*k + 3)) / \
    #         (1 - Q**2)**2
    #     second = 1/sigmasq * RinvV @ np.ones(N) * \
    #         1 / (1-Q)**2 * (1 - Q**2) / Q**(2*k + 2)
    #     third = 1/sigmasq * V.T @ RinvV * \
    #         (Q**(2*k + 3) - (k+1) * Q**(2*k + 1) * (1 - Q**2)) / \
    #         Q**(4*k + 4)

    #     return first + second + third

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

        return -N/2 * 1/sigmasq + 1/2 * 1/sigmasq**2 * V.T @ RinvV

    def dCidMu(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Ci = Xs[:, i] / Q**(i+2)

        V = Ci - mu
        RinvV = chol_solve((Rchol, True), V)

        return 1/sigmasq * RinvV @ np.ones(N)

    def dCidQ(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Ci = Xs[:, i] / Q**(i+2)

        V = Ci - mu
        RinvV = chol_solve((Rchol, True), V)

        return - 1 / sigmasq * RinvV @ Ci/Q * (i+2)

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
        return (-alpha_sig - 1) / sigmasq + beta_sig / sigmasq**2

    def dJointdQ(position):
        # return dDeltadQ(position) + dCdQ(position) + dlogQdQ(position)
        return dCdQ(position) + dlogQdQ(position) + dJacobiandQ(position)

    def dJointdMu(position):
        # return dDeltadMu(position) + dCdMu(position) + dlogMudMu(position)
        return dCdMu(position) + dlogMudMu(position)

    def dJointdSigmasq(position):
        # return dDeltadSigmasq(position) + dCdSigmasq(position) + \
        #     dlogSigmasqdSigmaSq(position)
        return dCdSigmasq(position) + dlogSigmasqdSigmaSq(position)

    def gradU(position):
        return np.array([dJointdQ(position), dJointdMu(position),
                         dJointdSigmasq(position)])

    return gradU

A_data = pd.read_csv("./data/A_data.csv")
A_data = A_data[A_data.Energy == 100]
theory_points = A_data.loc[:, ['2', '3', '4', '5']].values
kinpars = A_data.theta.values
jitter = 1e-7
length_scales = 30  # TOTALLY MADE UP

U_function = UFactory(
    theory_points, kinpars, length_scales, jitter, mu_mu=0, sigmasq_mu=10,
    alpha_Q=1, beta_Q=1, alpha_sig=1, beta_sig=1)
grad_U_function = GradUFactory(
    theory_points, kinpars, length_scales, jitter, mu_mu=0, sigmasq_mu=10,
    alpha_Q=1, beta_Q=1, alpha_sig=1, beta_sig=1)

bayes_model = HamiltonianSampler(U_function, grad_U_function, num_leaps=50, step_size=0.00001, tempering=np.array([0.6, 1.01, 1.01]))

start_position = np.array([0.6, 0.05, 1])  # Q, Mu, SigamSq

bayes_model.initialize(start_position)
bayes_model.set_bounds({0: (0, 1),
                        1: (-np.Infinity, np.Infinity),
                        2: (0, np.Infinity)})
bayes_model.set_seed(2343)
#bayes_model.burn_in(100)
results = bayes_model.sample(200)

Q = results[:, index["Q"]]
mu = results[:, index["MU"]]
sigmasq = results[:, index["SIGMASQ"]]

delta_mu = Q**6 / (1 - Q) * mu
delta_sig = np.sqrt(Q**12 / (1 - Q**2) * sigmasq)
deltas = np.random.normal(loc=delta_mu, scale=delta_sig)

print_res = np.zeros((results.shape[0], results.shape[1] + 2))
print_res[:, 0:5] = results
print_res[:, 3] = delta_mu
print_res[:, 4] = delta_sig

plt.plot(Q)
plt.show()

print(results) #print_res)

print("Acceptance Percentage: ", bayes_model.number_accepted / bayes_model.total_number_of_draws)
