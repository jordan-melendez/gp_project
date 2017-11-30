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
        return (alpha_Q - 1) * np.log(Q) + (beta_Q - 1) * np.log(1 - Q)

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

    N, K = Xs.shape

    Rinv = None # Doesn't depend on anything by the Xs and the rhos, so its constant wrt Q, sigsq, mu
                # LogDetR doesn't need calculated since its constant and cancels in MH step

    def dDeltadSigmasq(position):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        
        V = delta - mu / (1-Q)

        return -N/2 * 1/sigmasq + 1/2 * 1/sigmasq^2 * V @ Rinv @ V * (1 - Q^2) / Q^(2*(K+2 + 1))

    def dDeltadMu(position):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)

        V = delta - mu / (1-Q)

        # TODO Double check the value K, K+2, etc.
        return 1/sigmasq * Rinv @ V * 1/(1-Q)^2 * (1-Q^2)/Q^(2*(K+2 + 1))

    def dDeltadQ(position):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)

        k = K + 2

        V = delta - mu / (1-Q)

        first = (1-Q^2)/Q^(2*k+2) * (2 * (k+1) * Q^(2*k + 1) * (1-Q^2) + 2*Q^(2*k + 3)) / (1 - Q^2)^2
        second = 1/sigmasq * Rinv @ V @ np.ones(N) * 1 / (1-Q)^2 * (1 - Q^2) / Q^(2*k + 2)
        third = 1/sigmasq * V.T @ Rinv @ V * (Q^(2*k + 3) - (k+1) * Q^(2*k + 1) * (1 - Q^2)) / Q^(4*k + 4)

        return first + second + third


    def dCidSigmasq(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Ci = Xs[:, i] / Q^(i+2)

        V = Ci - mu

        return -N/2 * 1/sigmasq + 1/2 * 1/sigmasq^2 * V.T @ Rinv @ V

    def dCidMu(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Ci = Xs[:, i] / Q^(i+2)

        V = Ci - mu

        return 1/sigmasq * Rinv @ V @ np.ones(N)

    def dCidQ(position, i):
        Q = position[index["Q"]]
        sigmasq = position[index["SIGMASQ"]]
        mu = position[index["MU"]] * np.ones(N)
        Ci = Xs[:, i] / Q^(i+2)

        V = Ci - mu

        return 1 / sigmasq * Rinv @ V @ Ci/Q * (i+2)

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
        return -(alpha_sig - 1) / sigmasq + beta_sig / sigmasq^2

    def dJointdQ(position):
        return dDeltadQ(position) + dCdQ(position) + dlogQdQ(position)

    def dJointdMu(position):
        return dDeltadMu(position) + dCdMu(position) + dlogMudMu(position)
    
    def dJointdSigmasq(position):
        return dDeltadSigmasq(position) + dCdSigmasq(position) + dlogSigmasqdSigmaSq(position)

    def gradU(position):
        return np.array([dJointdQ(position), dJointdMu(position), dJointdSigmasq(position)])

    return gradU
