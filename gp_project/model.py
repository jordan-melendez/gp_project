import numpy as np

index = {
    "Q": 0,
    "MU": 1,
    "SIGMASQ": 2
}

def UFactory(delta, Xs, rhos, mu_mu, sigmasq_mu, alpha_Q, beta_Q, alpha_sig, beta_sig):

    N, K = Xs.shape

    Rinv = None # Doesn't depend on anything by the Xs and the rhos, so its constant wrt Q, sigsq, mu
                # LogDetR doesn't need calculated since its constant and cancels in MH step

    def logDelta(position):
        sigsq = position[index["SIGMASQ"]]
        mu = position[index["MU"]]
        Q = position[index["Q"]]

        V = delta - mu / (1 - Q)

        return -N/2 * np.log(sigsq) \
               - 1/2 * logDetR(position) \
               - N/2 * Q^(2*K + 2) / (1 - Q^2) \
               - 1/(2*sigsq) * V.T @ Rinv @ V * (1-Q^2) / Q^(2*K + 2)

    def logCi(position, i):
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Q = position[index["Q"]]
        # Since index starts at 0, we need to add 2 to get the correct order
        Ci = Xs[:, i] / Q^(i + 2)

        V = Ci - mu

        return -N/2 * np.log(sigmasq) - 1/(2 * sigmasq)  * V.T @ Rinv @ V

    def logC(position):
        return sum(logCi(position, i) for i in range(K))

    def logQ(position):
        Q = position[index["Q"]]
        return (alpha - 1) * np.log(Q) + (beta - 1) * np.log(1 - Q)

    def logMu(position):
        mu = position[index["MU"]]
        return -1/(2 * sigmasq_mu) * (mu - mu_mu)^2

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

def GradUFactory(delta, Xs, rhos, mu_mu, sigmasq_mu, alpha_sig, beta_sig, alpha_Q, beta_Q):

    def Rinverse(position):
        raise NotImplementedError

    def dDeltadSigmasq(position):
        raise NotImplementedError

    def dDeltadMu(position):
        raise NotImplementedError

    def dDeltadQ(position):
        raise NotImplementedError

    def dCidSigmasq(position):
        raise NotImplementedError

    def dCidMu(position):
        raise NotImplementedError

    def dCidQ(position):
        raise NotImplementedError

    def dCdSigmasq(position):
        raise NotImplementedError

    def dCdMu(position):
        raise NotImplementedError

    def dCdQ(position):
        raise NotImplementedError

    def dlogQdQ(position):
        Q = position[index["Q"]]
        return (alpha - 1) / Q - (beta - 1) / (1 - Q)

    def dlogMudMu(position):
        raise NotImplementedError

    def dlogSigmasqdSigmaSq(position):
        raise NotImplementedError

    def dJointdQ(position):
        return dDeltadQ(position) + dCdQ(position) + dlogQdQ(position)

    def dJointdMu(position):
        return dDeltadMu(position) + dCdMu(position) + dlogMudMu(position)
    
    def dJointdSigmasq(position):
        return dDeltadSigmasq(position) + dCdSigmasq(position) + dlogSigmasqdSigmaSq(position)

    def gradU(position):
        return np.array([dJointdQ(position), dJointdMu(position), dJointdSigmasq(position)])

    return gradU
