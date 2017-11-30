import numpy as np
import scipy as sp
from scipy import cholesky
from functools import partial
from .kronecker import gaussian_kernel

chol_solve = partial(sp.nlinalg.cho_solve, check_finite=False)

index = {
    "Q": 0,
    "MU": 1,
    "SIGMASQ": 2
}


def UFactory(delta, Xs, kinpars, ls, jitter, mu_mu, sigmasq_mu, alpha_sig,
             beta_sig, alpha_Q, beta_Q):

    N, K = Xs.shape
    R = gaussian_kernel(kinpars, kinpars, ls) + jitter * np.eye(N)
    Rchol = np.linalg.cholesky(R)

    def logDelta(position):
        sigsq = position[index["SIGMASQ"]]
        mu = position[index["MU"]]
        Q = position[index["Q"]]

        V = delta - mu / (1 - Q)
        RinvV = chol_solve((Rchol, True), V)

        return -N/2 * np.log(sigsq) \
               - N/2 * Q**(2*K + 2) / (1 - Q**2) \
               - 1/(2*sigsq) * V.T @ RinvV * (1-Q**2) / Q**(2*K + 2)

    def logCi(position, i):
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Q = position[index["Q"]]
        # Since index starts at 0, we need to add 2 to get the correct order
        Ci = Xs[:, i] / Q**(i + 2)

        V = Ci - mu
        RinvV = chol_solve((Rchol, True), V)

        return -N/2 * np.log(sigmasq) - 1/(2 * sigmasq) * V.T @ RinvV

    def logC(position):
        return sum(logCi(position, i) for i in range(K))

    def logQ(position):
        Q = position[index["Q"]]
        return (alpha_Q - 1) * np.log(Q) + (beta_Q - 1) * np.log(1 - Q)

    def logMu(position):
        mu = position[index["MU"]]
        return -1/(2 * sigmasq_mu) * (mu - mu_mu)**2

    def logSigmasq(position):
        sigmasq = position[index["SIGMASQ"]]
        return (alpha_sig - 1) * np.log(sigmasq) - beta_sig * sigmasq

    def U(position):
        return logDelta(position) + \
               logC(position) + \
               logQ(position) + \
               logMu(position) + \
               logSigmasq(position)

    return U


def GradUFactory(delta, Xs, kinpars, ls, jitter, mu_mu, sigmasq_mu, alpha_sig,
                 beta_sig, alpha_Q, beta_Q):

    N, K = Xs.shape
    R = gaussian_kernel(kinpars, kinpars, ls) + jitter * np.eye(N)
    Rchol = np.linalg.cholesky(R)

    def dDeltadSigmasq(position):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)

        V = delta - mu / (1-Q)
        RinvV = chol_solve((Rchol, True), V)

        return -N/2 * 1/sigmasq + 1/2 * 1/sigmasq**2 * V @ RinvV * \
            (1 - Q**2) / Q**(2*(K+2 + 1))

    def dDeltadMu(position):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)

        V = delta - mu / (1-Q)
        RinvV = chol_solve((Rchol, True), V)

        # TODO Double check the value K, K+2, etc.
        return 1/sigmasq * RinvV * 1/(1-Q)**2 * (1-Q**2)/Q**(2*(K+2 + 1))

    def dDeltadQ(position):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)

        k = K + 2

        V = delta - mu / (1-Q)
        RinvV = chol_solve((Rchol, True), V)

        first = (1-Q**2)/Q**(2*k+2) * \
            (2 * (k+1) * Q**(2*k + 1) * (1-Q**2) + 2*Q**(2*k + 3)) / \
            (1 - Q**2)**2
        second = 1/sigmasq * RinvV @ np.ones(N) * \
            1 / (1-Q)**2 * (1 - Q**2) / Q**(2*k + 2)
        third = 1/sigmasq * V.T @ RinvV * \
            (Q**(2*k + 3) - (k+1) * Q**(2*k + 1) * (1 - Q**2)) / \
            Q**(4*k + 4)

        return first + second + third

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

        return 1 / sigmasq * RinvV @ Ci/Q * (i+2)

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
        return - (mu - mu_mu)

    def dlogSigmasqdSigmaSq(position):
        sigmasq = position[index["SIGMSQ"]]
        return -(alpha_sig - 1) / sigmasq + beta_sig / sigmasq**2

    def dJointdQ(position):
        return dDeltadQ(position) + dCdQ(position) + dlogQdQ(position)

    def dJointdMu(position):
        return dDeltadMu(position) + dCdMu(position) + dlogMudMu(position)

    def dJointdSigmasq(position):
        return dDeltadSigmasq(position) + dCdSigmasq(position) + dlogSigmasqdSigmaSq(position)

    def gradU(position):
        return np.array([dJointdQ(position), dJointdMu(position), dJointdSigmasq(position)])

    return gradU
