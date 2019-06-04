__author__ = 'Ziyan Zhu'

from sklearn.linear_model import LassoCV,LogisticRegressionCV
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import numpy as np
from statsmodels.distributions import ECDF
from scipy.stats import norm

# http://arxiv.org/abs/1202.1377 P.Buehlmann
def switch_binomial(X,y):
    n, p =X.shape
    fitnet = LogisticRegressionCV(cv=10,n_jobs=-1)
    fitnet.fit(X,y)
    pihat = fitnet.predict_proba(X)
    betahat = fitnet.coef_

    diagW = pihat*(1-pihat)
    W = np.diag(diagW)
    xl = np.column_stack((np.ones(n),X))

    # Adjusted design matrix
    xw = np.sqrt(diagW) * X

    # Adjusted response
    yw = np.sqrt(diagW) * (np.dot(xl,betahat)+np.linalg.solve(W,y-pihat))

    return xw,yw

class Ridge_Proj:

    def __init__(self,standardize=True,ridge_unprojected=False,family="gaussian"):
        self.family = family
        self.standardize = standardize
        self.ridge_unprojected = ridge_unprojected

    def pval_adjust_WY(self, cov, pval, N=10000):
        ## Purpose:
        ## multiple testing correction with a Westfall young-like procedure as
        ## in ridge projection method, http://arxiv.org/abs/1202.1377 P.Buehlmann
        ## ----------------------------------------------------------------------
        ## Arguments:
        ## cov: covariance matrix of your estimator
        ## pval: the single testing pvalues
        ## N: the number of samples to take for the empirical distribution
        ##    which is used to correct the pvalues
        ## ----------------------------------------------------------------------
        ## Author: Ruben Dezeure, Date: 6 Feb 2014, 14:27
        ncol = cov.shape[1]
        zz = np.random.multivariate_normal(mean=np.zeros(ncol), cov=cov, size=N)
        zz2 = zz / np.sqrt(np.diagonal(cov))
        gz = 2 * norm.sf(abs(zz2))
        GZ = np.min(gz, axis=0)

        ecdf = ECDF(GZ)
        pcorr = ecdf(pval)
        return pcorr

    def tsvd(self,A):
        n,p = A.shape
        tol = min(n, p) * np.finfo(np.float64).eps
        tsvd = TruncatedSVD(n_components=min(n,p),tol=tol)
        tsvd.fit(A)

        Sval = tsvd.singular_values_

        rankA = np.count_nonzero(Sval>=(tol*Sval[0]))

        u,s,vh = np.linalg.svd(A,full_matrices=True)
        return u[:,:rankA],s[:rankA].reshape((rankA,1)),vh.T[:,:rankA]

    def fit(self,X,y,lam=1):
        n,p = X.shape

        if self.standardize:
            sds = np.std(X, axis=0, ddof=1)
        else:
            sds = np.ones(p)

        # center the columns to get rid of the intercept#
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0, ddof=1))

        if self.family == "binomial":
            X, y = switch_binomial(X, y)

        X = (X - np.mean(X, axis=0))
        y = (y - np.mean(y, axis=0))

        # SVD
        U,S,V = self.tsvd(X)

        Px = np.dot(V,V.T)
        Px_offdiag = Px.copy()
        np.fill_diagonal(Px_offdiag,0)
        ## Use svd for getting the inverse for the ridge problem

        Omiga = np.dot(V,(S / (S**2 +lam))* U.T)

        ## Ruben Note: here the S^2/n 1/n factor has moved out, the value of lambda
        ## used is 1/n! See also comparing to my version

        cov2 = np.dot(Omiga,Omiga.T)
        diag_cov2 = np.diagonal(cov2)

        # get initial estimators with LassoCV
        model = LassoCV(cv=10, n_jobs=-1).fit(X, y)  # use 3 CPUs
        betainit = model.coef_
        hy = model.predict(X)
        hsigma = np.sqrt(np.sum(np.square(y - hy)) / (n - np.sum(betainit != 0)))

        hsigma2 = hsigma**2

        ## bias correction
        biascorr = np.dot(Px_offdiag.T,betainit)

        ## ridge estimator
        hbeta = np.dot(Omiga ,y)

        hbetacorr = hbeta - biascorr

        if self.ridge_unprojected:  ## bring it back to the original scale
            hbetacorr = hbetacorr / np.diagonal(Px)


        ## Ruben Note: a_n = 1 / scale.vec, there is no factor sqrt(n) because this
        ## falls away with the way diag.cov2 is calculated see paper
        scale_vec = np.sqrt(hsigma2 * diag_cov2)

        if self.ridge_unprojected:
            scale_vec = scale_vec/abs(np.diagonal(Px))

        hbetast = hbetacorr/scale_vec

        Delta = (1/scale_vec) * np.max(abs(Px_offdiag),axis=0)*((np.log10(p)/n)**0.45)

        if (self.ridge_unprojected): ## 2
            Delta = Delta / abs(np.diag(Px))

        hgamma = abs(hbetast)

        #########################
        ## Individual p-values ##
        #########################
        temp = 2 * norm.sf(np.abs(hgamma-Delta))

        res_pval = np.minimum(temp,1)
        self.pvals = res_pval

        #########################################
        ## Multiple testing corrected p-values ##
        #########################################

        self.pvals_corr = self.pval_adjust_WY(cov2,res_pval)
