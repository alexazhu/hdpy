__author__ = 'Ziyan Zhu, Haohan Wang'

import numpy as np
import math
from sklearn.linear_model import Lasso,  lasso_path, LassoCV, RidgeCV, lasso_path, LassoLars, LogisticRegressionCV
from statsmodels.api import OLS
import sys
from statsmodels.distributions import ECDF
from scipy.stats import norm, t
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
#import seaborn as sns
import matplotlib.pyplot as plt
from multisplit import Multisplit
from debiasedLasso_skLearn import DebiasedLasso
from ldpe import Lassoproj
from ridgeproj import Ridgeproj

def switch_binomial(X, y):
    n, _ = X.shape
    fitnet = LogisticRegressionCV(cv=10, n_jobs=-1)
    fitnet.fit(X, y)
    pihat = fitnet.predict_proba(X)
    betahat = fitnet.coef_

    diagW = pihat * (1 - pihat)
    W = np.diag(diagW)
    xl = np.column_stack((np.ones(n), X))

    # Adjusted design matrix
    xw = np.sqrt(diagW) * X

    # Adjusted response
    yw = np.sqrt(diagW) * (np.dot(xl, betahat) + np.linalg.solve(W, y - pihat))

    return xw, yw

def pval_adjust_WY(self, cov, pvals, N=10000):
        """
        Purpose:
        multiple testing correction with a Westfall young-like procedure as
        in ridge projection method, http://arxiv.org/abs/1202.1377 P.Buehlmann
        ======================================================================
        :param cov: covariance matrix of your estimator
        :param pvals: single testing pvalues
        :param N: the number of samples to take for the empirical distribution
        :return pcorr: corrected p-values
        ======================================================================
        Author: Ziyan Zhu, Date: April 10th, 2019
        Following R version by Ruben Dezeure, Date: 6 Feb 2014, 14:27
        """
        ncol = cov.shape[1]
        zz = np.random.multivariate_normal(mean=np.zeros(ncol), cov=cov, size=N)
        zz2 = zz / np.sqrt(np.diagonal(cov))
        gz = 2 * norm.sf(abs(zz2))
        GZ = np.min(gz, axis=0)

        ecdf = ECDF(GZ)
        pcorr = ecdf(pvals)
        return pcorr

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

def hd(method="multi-split", X=None, y=None, family="gaussian",
       ci=False, ci_level=0.95, B=100, fraction=0.5, repeat_max=20, manual_lam=True,
       aggr_ci=False, return_nonaggr=False,
       model_selector=LassoCV(cv=10, n_jobs=-1, fit_intercept=False, tol=0.0001),
       robust=False, standardize=True,
       ridge_unprojected=False):
    """
        :param method: Inference methods: multi-split,lasso-proj,ridge-proj,debiasedLasso
        :param X: Input matrix
        :param y: Output vector
        :param family: Binomial or gaussian
        :param ci: Calculate confidence intervals or not
        :param ci_level: Confidence level

        ===== Multi-split =====
        :param B: repeat times
        :param fraction: sample splitting
        :param repeat_max: maximum repeat times
        :param model_selector: model for variable selection
        :param manual_lam: 'True' calculate regularization parameters by hand
        :param aggr_ci: 'True' aggregation on confidence intervals; 'False' take median instead
        :param return_nonaggr: return p-values matrix before aggregation

        ===== Lasso-proj =====
        :param standardize: 'True' standardize input data
        :param intercept: 'True' include intercept.
        :param robust: 'True' use robust sigma

        ===== Ridge-proj =====
        :param ridge_unprojected: 'True' return unprojected results

        """

    if method == "multi-split" or method == "multi":
        est = Multisplit(ci=ci, ci_level=ci_level, B=B, fraction=fraction, manual_lam=manual_lam, aggr_ci=aggr_ci,
                          return_nonaggr=return_nonaggr, repeat_max=repeat_max)
        est.fit(X,y)
        pvalues = est.pvals_corr
        
    if method == "lasso-proj" or method == "lassoproj":
        est = Lassoproj(ci=ci, ci_level=ci_level, standardize=standardize, robust=robust, family=family)
        est.fit(X,y)
        pvalues = est.pvals_corr

    if method == "ridge-proj" or method == "ridgeproj":
        est = Ridgeproj(standardize=standardize, ridge_unprojected=ridge_unprojected, family=family)
        est.fit(X,y)
        pvalues = est.pvals_corr

    if method == "debiased-lasso" or method == "debiased":
        est = DebiasedLasso()
        est.fit(X,y)
        pvalues = est.pvals
    
    print("Getting corrected p-values")
    print(np.argwhere(pvalues<0.05))
    return pvalues

