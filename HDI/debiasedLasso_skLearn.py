__author__ = 'Haohan Wang, Ziyan Zhu'

from sklearn.linear_model import LassoCV, Lasso
import numpy as np
from scipy.stats import norm


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        implementation from: https://stackoverflow.com/a/23535934/1995263
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    result = 1.4826 * np.median(np.abs(arr - med))
    return result, med

class debiasedLasso:
    def __init__(self):
        self.model = LassoCV()

    def noiseEstimator(self, b, A, n):
        ynorm = np.sqrt(n) * (b / np.sqrt(np.diag(A)))
        sd_hat0, med = mad(ynorm)
        zeros = np.abs(ynorm) - med < 3 * sd_hat0
        y2norm = np.sum(b[zeros == 1] ** 2)
        Atrace = np.sum(np.diag(A)[zeros == 1])
        sd_hat1 = np.sqrt(n * y2norm / Atrace)
        # skip the checking whether the noise is problematic step
        s0 = np.sum(zeros == 0)

        return sd_hat1, s0

    def inverseLinftyOneRow(self,sigma, i, mu, maxiter=50, threshold=1e-2):
        p = sigma.shape[0]
        if i == 0:
            rho = np.max(np.abs(sigma[i, i + 1:])) / sigma[i, i]
        elif i == p - 1:
            rho = np.max(np.abs(sigma[i, :i])) / sigma[i, i]
        else:
            rho = max(np.max(np.abs(sigma[i, :i])), np.max(np.abs(sigma[i, i + 1:]))) / sigma[i, i]
        mu0 = rho / (1 + rho)

        beta = np.zeros(p)

        if mu >= mu0:
            beta[i] = (1 - mu0) / sigma[i, i]
            return beta, 0

        diff_norm2 = 1
        last_norm2 = 1
        iter = 1
        iter_old = 1
        beta[i] = (1 - mu0) / sigma[i, i]

        beta_old = beta.copy()
        sigma_tilde = sigma.copy()
        np.fill_diagonal(sigma_tilde, 0)
        vs = -np.dot(sigma_tilde, beta)

        while (iter <= maxiter) and (diff_norm2 >= (threshold * last_norm2)):
            for j in range(p):
                oldval = beta[j]
                v = vs[j]
                if j == i:
                    v = v + 1
                beta[j] = (np.sign(v) * max(0, abs(v) - mu)) / sigma[j, j] # Soft thresholding
                if oldval != beta[j]:
                    vs = vs + (oldval - beta[j]) * sigma_tilde[:, j]

            iter = iter + 1
            if iter == 2 * iter_old:
                d = beta - beta_old
                diff_norm2 = np.sqrt(sum(d * d))
                last_norm2 = np.sqrt(sum(beta * beta))
                iter_old = iter
                beta_old = beta.copy()
                if iter > 10:
                    vs = -np.dot(sigma_tilde, beta)

        return beta, iter

    def inverseLinfty(self,sigma, n, resol=1.5, mu=None, maxiter=50, threshold=1e-2):
        if mu is None:
            isgiven = 0
        else:
            isgiven = 1

        p = sigma.shape[0]
        M = np.zeros(shape=[p, p])

        beta = np.zeros(p)

        for i in range(p):
            if isgiven == 0:
                mu = (1 / np.sqrt(n)) * norm.ppf(1 - (0.1 / (p ** 2)))
            mu_stop = 0
            try_no = 1
            incr = 0
            while (mu_stop != 1) and try_no < 10:
                last_beta = beta.copy()
                beta, iter = self.inverseLinftyOneRow(sigma, i, mu, maxiter=maxiter, threshold=threshold)
                if isgiven == 1:
                    mu_stop = 1
                else:
                    if try_no == 1:
                        if iter == maxiter + 1:
                            incr = 1
                            mu = mu * resol
                        else:
                            incr = 0
                            mu = mu / resol
                    if try_no > 1:
                        if incr == 1 and iter == maxiter + 1:
                            mu = mu * resol
                        if incr == 1 and iter < maxiter + 1:
                            mu_stop = 1
                        if incr == 0 and iter < maxiter + 1:
                            mu = mu / resol
                        if incr == 0 and iter == maxiter + 1:
                            mu = mu * resol
                            beta = last_beta.copy()
                            mu_stop = 1
                try_no = try_no + 1

            M[i, :] = beta
        return M

    def binarySearch(self, X, y, select_num = 50):
        model = Lasso()
        betaM = np.zeros([X.shape[1]])
        #max(abs(crossprod(xx, yy / sqrt(sum(yy ^ 2)) / sqrt(n))))
        iteration = 0
        min_lambda = 1e-15
        max_lambda = 1e15

        minFactor = 0.9
        maxFactor = 1.1

        stuckCount = 1
        previousC = -1

        patience = 20

        while min_lambda < max_lambda and iteration < 50:
            iteration += 1
            lmbd = np.exp((np.log(min_lambda) + np.log(max_lambda)) / 2.0)

            # print "\t\tIter:{}\tlambda:{}".format(iteration, lmbd),
            model.set_params(alpha=lmbd)
            model.fit(X, y)
            beta = model.coef_

            c = len(np.where(np.abs(beta) > 0)[0])  # we choose regularizers based on the number of non-zeros it reports
            # print "# Chosen:{}".format(c)
            if c < select_num * minFactor:  # Regularizer too strong
                max_lambda = lmbd
                betaM = beta
            elif c > select_num * maxFactor:  # Regularizer too weak
                min_lambda = lmbd
                betaM = beta
            else:
                betaM = beta
                break
            if c == previousC:
                stuckCount += 1
            else:
                previousC = c
                stuckCount = 1
            if stuckCount > patience:
                # print 'Run out of patience'
                break

        return betaM

    def fit(self, X, y, alpha=None, intercept=False):
        """

        :param X: design matrix
        :param y: response
        :param intercept:
        :param lmbd: Lasso regularization parameter (if null, fixed by sqrt lasso)

        :return:
            p-values, confidence intervals
        """

        print ("preparing")
        n,p= X.shape
        pp = p
        col_norm = 1.0 / np.sqrt((1.0 / n) * np.diag(np.dot(X.T, X)))
        X = np.dot(X, np.diag(col_norm))

        if alpha == None:
            alpha = np.sqrt(norm.ppf(1-(0.1/p))/n)

        # Objective : sqrt(RSS/n) +lambda *penalty

        #sigma_hat_diag = np.zeros(p)
        # col_norm = np.ones(p)
        # for i in range(p):
        # sigma_hat_diag[i] = (1.0 / n) * np.dot(X[:, i].T, X[:, i])
        # if sigma_hat_diag[i] != 0:
        # col_norm[i] = 1.0 / np.sqrt(sigma_hat_diag[i])
        # X[:, i] = X[:, i] * col_norm[i]
        # else:
        # s = np.sum(X[:, i])
        # if s != 0:
        # X[:, i] = X[:, i] / s



        if intercept == True:
            Xb = np.concatenate(np.ones(n),X)
            col_norm = np.concatenate(1,col_norm)
            pp = p + 1
        else:
            Xb = X

        sigma_hat = (1.0 / n) * np.dot(Xb.T, Xb)

        # X2 = np.zeros(p)
        # for i in range(p):
        # X2[i] = np.dot(X[:, i], X[:, i].T)

        if n>=2*p:
            tmp = np.linalg.eig(sigma_hat)
            tmp = np.min(tmp)/np.max(tmp)
        else:
            tmp = 0

        if n>=2*p and tmp>=1e-4:
            M = np.linalg.inv(sigma_hat)
        else:
            M = self.inverseLinfty(sigma=sigma_hat, n=n, resol=1.3, mu=None, maxiter=50, threshold=1e-2)

        A = np.dot(np.dot(M, sigma_hat), M.T)

        print('fitting')
        # self.model.fit(X, y)
        # beta = self.model.coef_
        #hbeta =self.binarySearch(X, y)
        thetamodel = LassoCV(fit_intercept=intercept)
        thetamodel.fit(X,y)
        htheta = thetamodel.coef_

        print('denoising')
        r = y.reshape((n,)) - np.dot(Xb, htheta)

        unbiased_lasso = htheta + np.dot(np.dot(M, Xb.T), r) / n
        # beta = beta + 1.0/n*np.dot(X.T, r)
        unbiased_lasso = unbiased_lasso.reshape((p,))

        sdhat, nz = self.noiseEstimator(unbiased_lasso, A, n)
        unbiased_lasso = unbiased_lasso * col_norm
        ps = 2 * (1 - norm.cdf(np.sqrt(n) * abs(unbiased_lasso) / (sdhat * col_norm * np.sqrt(np.diag(A)))))

        self.pvals = ps
        self.bhat = unbiased_lasso
