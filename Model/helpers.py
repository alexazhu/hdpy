__author__ = 'Ziyan Zhu'

from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso,LassoCV,LassoLars,lasso_path
import numpy as np
import math
from scipy.stats import norm
from scipy.interpolate import interp1d
from statsmodels.distributions import ECDF


def nodewiselasso(self, x, y, lambdaseq="quantile", lambdatuningfactor=1):
    # Find a good score vector zj to approximate the weight vector wj
    n, p = x.shape

    # Get a sequence of lambda values over all regressions for furthur selection
    # According to Friedman, Hastie & Tibshirani (2010) 'strategy is to select a minimum value lambda_min = epsilon * lambda_max, and construct a sequence of K values of lambda decreasing from lambda_max to lambda_min on the log scale. Typical values are epsilon = 0.001 and K = 100.'


    lams = np.array([])
    lambdas = np.zeros(100)
    for c in range(p):
        temp = np.copy(x)
        temp = np.delete(temp, c, axis=1)
        lam, _, dualgaps = lasso_path(X=temp, y=x[:, c], n_alphas=100)
        lams = np.append(lams, lam)

    lams = np.sort(lams)

    if lambdaseq == "quantile":
        ## Equidistant quantiles of the complete set of lambda values are returned.
        seq = np.linspace(0, 100, 100, dtype=int)
        lambdas = -np.sort(-lams[seq])

    if lambdaseq == "linear":
        ## A linear interpolation of lambda values between the max and min lambda values found.
        lambdas = np.arange(start=np.min(lams), stop=np.max(lams), step=(np.max(lams) - np.min(lams)) / 100)
        lambdas = -np.sort(-lambdas)

    ## Find the desired lambda:
    #     Use 10-fold cv to find the best lambda with the minimized error.

    bestlambda = self.cv_bestlambda(lambdas, x, K=10, lambdatuningfactor="lambda.min")

    Z = np.zeros(shape=(n, p))

    getZ = Lasso(alpha=bestlambda, max_iter=100)

    for i in range(p):
        temp = np.copy(x)
        temp = np.delete(temp, i, axis=1)
        getZ = getZ.fit(X=temp, y=x[:, i])
        prediction = getZ.predict(X=temp)
        Z[:, i] = x[:, i] - prediction

    ## rescale Z such that t(Zj) Xj/n = 1 for all j
    scaleZ = np.diag(np.dot(Z.T, x)) / n
    Z = np.true_divide(Z, scaleZ)

    return scaleZ, Z


def predictlars(betas, lam, s, X):
    k = betas.shape[0]
    s = np.where(s > np.max(lam), np.max(lam), s)
    s = np.where(s < 0, 0, s)
    sbeta = lam
    sfrac = (s - sbeta[0]) / (sbeta[k - 1] - sbeta[0])
    sbeta = (sbeta - sbeta[0]) / (sbeta[k - 1] - sbeta[0])

    usbeta = np.unique(sbeta)
    useq = (usbeta == sbeta)
    sbeta = sbeta[useq]
    betas = betas[useq, :]

    coord = interp1d(sbeta, np.array(range(len(sbeta))))
    coord = coord(sfrac)
    left = math.floor(coord)
    right = math.ceil(coord)

    newbetas = ((sbeta[right] - sfrac) * betas[left, :] + (sfrac - sbeta[left]) * betas[right, :]) / (
    sbeta[right] - sbeta[left])
    left = np.where(left == right, left, 0)
    newbetas = np.copy(betas[left, :])

    fit = np.dot(X, newbetas.T)
    return newbetas, fit

def sqrtlasso(X, y, intercept=False,alpha=None,n_alpha=None,rho=1,max_iter=50):
    n,p = X.shape
    maxdf = max(n, p)
    x1 = X - np.mean(X, axis=0)
    sdxinv = 1 / (np.sqrt(np.sum(x1 ** 2, axis=0) / (n - 1)))
    sdX = np.tile(sdxinv, (n, 1))
    xx = x1 * sdX
    yy = y - np.mean(y)
    sdy = 1

    if intercept:
        xx = np.column_stack((np.ones(xx.shape[0]), xx))
        X = np.column_stack((np.ones(n), X))
        p = p + 1

    if not alpha==None:
        n_alpha = len(alpha)
    else:
        n_alpha = 5
        if intercept:
            alpha_max = np.max(abs(np.dot(xx.T,yy/np.sqrt(np.sum(yy**2))/np.sqrt(n))))
        else:
            alpha_max = np.max(abs(np.dot(X.T,yy/np.sqrt(np.sum(yy**2))/np.sqrt(n))))



def scaledlasso(self, X, y, intercept, lam0=None, sigma=None):
        n, p = X.shape
        if lam0 == None:
            if p > pow(10, 6):
                lam0 = 'univ'
            else:
                lam0 = 'quantile'

        if lam0 == 'univ' or lam0 == 'universal':
            lam0 = np.sqrt(2 * np.log10(p) / n)

        if lam0 == 'quantile':
            L = 0.1
            Lold = 0
            while (np.abs(L - Lold) > 0.001):
                k = (L**4 + 2 * L**2)
                Lold = L
                L = -norm.ppf(np.min(k/p,0.99))
                L = (L + Lold) / 2
            if (p == 1):
                L = 0.5
            lam0 = np.sqrt(2 / n) * L

        sigmaint = 0.1
        sigmanew = 5
        flag = 0

        objlasso = LassoLars(fit_intercept=False,eps=0.001,fit_path=True)
        objlasso.fit(X,y)

        while abs(sigmaint - sigmanew) > 0.0001 and flag <= 100:
            flag = flag + 1
            sigmaint = np.copy(sigmanew)
            lam = lam0 * sigmaint
            s = lam * n
            lams = objlasso.alphas_
            s[np.where(s>np.max(lams))[0]]=np.max(lams)
            s[np.where(s<0)[0]]=0

            sfrac = (s-s[0])/(s[p-1]-s[0])
            s = (s-s[0])/(s[p-1]-s[0])


            hbeta = objlasso.coef_

            hy = np.dot(X,hbeta)
            sigmanew = np.sqrt(np.mean(np.square(y - hy)))

        sigmahat = sigmanew
        hlam = lam

        if sigma == None:
            sigmahat = np.sqrt(np.sum(np.square(y - hy)) / (n - np.sum(hbeta != 0)))

        return hbeta, sigmahat
