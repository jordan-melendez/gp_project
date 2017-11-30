import numpy as np

index = {
    "Q": 0,
    "MU": 1,
    "SIGMASQ": 2
}

def UFactory(delta, Xs, rhos, prior_mu, prior_sigmasq, alpha, beta):

    N, K = Xs.shape

    def Rinverse(position):
        raise NotImplementedError

    def logDetR(position):
        raise NotImplementedError

    def logDelta(position):
        sigsq = position[index["SIGMASQ"]]
        mu = position[index["MU"]]
        Q = position[index["Q"]]
        Rinv = Rinverse(position)

        V = delta - mu / (1 - Q)

        return -N/2 * np.log(sigsq) \
               - 1/2 * logDetR(position) \
               - N/2 * Q^(2*K + 2) / (1 - Q^2) \
               - 1/(2*sigsq) * V.T @ Rinverse(position) @ V * (1-Q^2) / Q^(2*K + 2)

    def logCi(position):
        raise NotImplementedError

    def logC(position):
        raise NotImplementedError

    def logQ(position):
        Q = position[index["Q"]]
        return (alpha - 1) * np.log(Q) + (beta - 1) * np.log(1 - Q)

    def logMu(position):
        raise NotImplementedError

    def logSigmasq(position):
        raise NotImplementedError

    def U(position):
        return logDelta(position) * \
               logC(position) * \
               logQ(position) * \
               logMu(position) * \
               logSigmasq(position)

    return U

def GradUFactory(Xs, prior_mu, prior_sigmasq, alpha, beta):

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
