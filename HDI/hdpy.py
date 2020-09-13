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
import seaborn as sns
import matplotlib.pyplot as plt

def switch_binomial(X, y):
    n, p = X.shape
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


class multi_split:
    def __init__(self, ci=False, ci_level=0.95, B=100, fraction=0.5, repeat_max=20, manual_lam=True,
                 aggr_ci=True, return_nonaggr=False,
                 model_selector=LassoCV(cv=10, n_jobs=-1, fit_intercept=False, tol=0.001)):

        '''
        An implementation of Multiple Splitting methods https://doi.org/10.1198/jasa.2009.tm08647
        Following the R package hdi by Lukas Meier, Date:  2 Apr 2013, 11:52
        ==========================================================================================
        :param ci: calculate c.i. simultaneously
        :param ci_level: confidence level
        :param B: Times of single split
        :param fraction: the ratio of I1 to I2
        :param repeat_max: maximum repeat times
        :param model_selector: model for selection on the first half
        :param manul_lam: use manually calculated lambda sequences
        ==========================================================================================
        Attributes:
        p_nonaggr: Raw p-values that haven't been aggregated, shape of B*p
        pvals_corr: P-values aggregated controlling for family-wise error rate
        lci: Lower bounds of the confidence intervals
        uci: Upper bounds of the confidence intervals
        gamma_min: Smallest value of the gamma sequence
        ===========================================================================================
        Author: Ziyan Zhu, Date: Oct 4th,2019
        ==========================================================================================
        '''

        self.B = B
        self.ci = ci
        self.ci_level = ci_level
        self.fraction = fraction
        self.aggr_ci = aggr_ci
        self.return_nonaggr = return_nonaggr

        self.selector = model_selector
        self.manual_lam = manual_lam
        self.repeat_max = repeat_max
        self.gamma = np.arange(start=math.ceil(0.05 * B) / B, stop=1 - 1 / B, step=1 / B, dtype=float)

    def singlesplit(self, X, y, nleft):

        ######## One sample single split ########

        n, p = X.shape
        nright = n - nleft

        pvals_v = np.ones(p)
        lci_v = np.array([-np.Inf] * p)
        uci_v = np.array([np.Inf] * p)
        coefs = np.empty(p)
        ses_v = np.array([np.Inf] * p)

        tryagain = True
        count = 0

        while tryagain:

            ######## Randomly split the sample #######
            split = np.random.randint(low=1, high=n, size=nleft)  # without replacement
            xleft = X.copy()[split, :]
            yleft = y.copy()[split]

            xright = X.copy()[~split, :]
            yright = y.copy()[~split]

            ######## Model selection on Sample I #######

            if self.manual_lam:
                # calculate regularization path
                eps = 0.001
                K = 100
                max_lambda = np.max(np.abs(np.sum(np.dot(xleft.T, yleft)))) / n
                lambda_path = np.round(np.exp(np.linspace(math.log10(max_lambda), math.log10(max_lambda * eps), K)),decimals=100)
                self.selector.set_params(alphas=lambda_path)

            self.selector.fit(X=xleft, y=yleft)

            sel_nonzero = (self.selector.coef_ != 0)  # location of selected variables

            p_sel = sum(sel_nonzero)  # size of selected variables

            ######## Check up the selected results, make sure applicable for OLS ########

            if (p_sel + 1) >= nright:
                # rankX larger than number of row, cannot calculate p-values
                tryagain = True
                count = count + 1
                print("Too large model selected in a sample-split")

            if p_sel == 0:
                print("Empty model selected, it is OK")
                tryagain = False

            if p_sel > 0 and (p_sel + 1) < nright:

                tryagain = False

                ######## Fitting Sample II with reduced features using OLS ########

                lm = OLS(yright, xright[:, sel_nonzero]).fit(method="qr")

                df_res = lm.df_resid
                sel_pval = lm.pvalues

                coefs[sel_nonzero] = lm.params

                ses_v[sel_nonzero] = lm.bse

                # Sanity checks for p-values

                if len(sel_pval) != p_sel:
                    sys.exit(
                        "The statsmodels.OLS didn't return the correct number of p-values for the provided submodel.")
                if not (np.all(sel_pval >= 0) and np.all(sel_pval <= 1)):
                    sys.exit("The statsmodels.OLS returned p-values below 0 or above 1.")

                ######## Multiple testing adjustment on small sample: Bonferroni ########

                pvals_v[sel_nonzero] = np.minimum(sel_pval * p_sel, 1)  # renew p-values

                ######## Confidence intervals and other relative informations ########
                if all(pow(10, -5) < abs(self.gamma * self.B % 1)):
                    print("Duality might be violated because of choice of gamma. Use steps of length 1 / B")

                sel_ci = lm.conf_int(alpha=self.ci_level)

                lci_v[sel_nonzero] = sel_ci[:, 0]
                uci_v[sel_nonzero] = sel_ci[:, 1]

                ######## End of C.I. ########

            if count > self.repeat_max:
                print("Exceed max repeat times,sample splits resulted in too large models.")
                sys.exit()

        return pvals_v, p_sel, coefs, lci_v, uci_v, ses_v, df_res

    def fit(self, X, y):

        n, p = X.shape

        ######## Split the sample into two parts of approximately same size ########
        nleft = math.floor(n * self.fraction)
        nright = n - nleft

        if not (nleft >= 1 or nright >= 1):
            sys.exit("Not enough samples for splitting")

        ######## Repeat sample splitting for B times ########
        pvals = np.zeros((self.B, p))
        coefs = np.zeros((self.B, p))
        s0 = np.zeros(self.B)

        lci = np.zeros((self.B, p))
        uci = np.zeros((self.B, p))
        ses = np.zeros((self.B, p))
        df_res = np.zeros(self.B)

        h = Pool(2)
        for b in range(self.B):
            pvals[b, :], s0[b], coefs[b, :], lci[b, :], uci[b, :], ses[b, :], df_res[b] = h.apply_async(
                self.singlesplit, args=(X, y, nleft)).get()
        h.close()
        h.join()

        # for b in range(self.B):
        # pvals[b,:],coefs[b,:],lci[b,:],uci[b,:],ses[b,:],df_res[b]= self.singlesplit(X,y,nleft)

        if self.return_nonaggr:
            self.p_nonaggr = pvals

        ######## Aggregate into final p-values ########

        if not (0.05 in self.gamma):
            print("0.05 is not in gamma range due to the choice of B, the results might be incorrect.")

        if len(self.gamma) > 1:
            penalty = 1 - np.log10(np.min(self.gamma))
        else:
            penalty = 1

        quant_gamma = np.quantile(pvals, self.gamma, axis=0) / self.gamma.reshape(self.gamma.shape[0], 1)
        inf_quant = np.min(quant_gamma, axis=0)
        pvals_pre = inf_quant * penalty
        pvals_current = np.minimum(pvals_pre, 1)

        self.pvals_corr = pvals_current
        self.gamma_min = np.min(self.gamma)

        print("P-values Done")
        ######## Aggregate Confidence intervals

        if self.aggr_ci:
            self.s0 = 0
            self.d = p
            self.lci, self.uci = self.aggregate_ci(L=lci, U=uci, S0=s0, C=coefs, DF_RES=df_res, SES=ses)
        else:
            self.lci = np.median(lci, axis=1)
            self.uci = np.median(uci, axis=1)

    def does_it_cover(self, betaj, ci_info, multi_corr=False):
        alpha = 1 - self.ci_level

        ci_length = ci_info[2]
        no_inf_ci, center, ses, s0, df_res = ci_info[4:]
        # ci_Information = [lci, uci, ci_length, inf_ci, no_inf_ci, center, ses, s0, df_res]

        # the rank of the p-value in increasing order
        pval_rank = np.argsort(-np.true_divide(np.abs(betaj - center), ci_length / 2))
        # the number of ci + the inf ci we left out
        nsplit = len(pval_rank) + no_inf_ci

        gamma_b = pval_rank / nsplit

        if multi_corr:
            level = (1 - s0 * alpha * gamma_b / (1 - np.log(self.gamma_min)))
        else:
            level = (1 - alpha * gamma_b / (1 - np.log(self.gamma_min)))

        a = 1 - (1 - level) / 2

        if (gamma_b <= self.gamma_min).all():
            return True
        else:
            fac = t.ppf(a, df_res)

            nlci = (center - fac * ses)
            nuci = (center + fac * ses)

            position = (gamma_b > self.gamma_min)
            if (position).any():
                return (nlci[position] <= betaj).all() and (nuci[position] >= betaj).all()

    def find_bisection_bounds(self, shouldcover, shouldnotcover, ci_info):
        reset_shouldnotcover = False
        if self.does_it_cover(shouldnotcover, ci_info):
            reset_shouldnotcover = True
            while self.does_it_cover(shouldnotcover, ci_info):
                old = shouldnotcover.copy()
                shouldnotcover = shouldnotcover + (shouldnotcover - shouldcover)
                shouldcover = old.copy()
        if not self.does_it_cover(shouldcover, ci_info):
            if reset_shouldnotcover:
                sys.exit("Problem")
            while self.does_it_cover(shouldcover, ci_info):
                old = shouldcover.copy()
                shouldcover = shouldcover + (shouldcover - shouldnotcover)
                shouldnotcover = old.copy()

        return shouldcover, shouldnotcover

    def check_bounds(self, shouldcover, shouldnotcover, ci_info):
        if self.does_it_cover(shouldnotcover, ci_info):
            sys.exit("Shouldnotcover bound is covered")
        if not self.does_it_cover(shouldcover, ci_info):
            sys.exit("Shouldcover is a bad covered bound.")

    def bisection_gammamin_coverage(self, outer, inner, eps_bound, ci_info):
        self.check_bounds(inner, outer, ci_info)
        eps = 1
        while eps > eps_bound:
            middle = (outer + inner) / 2
            if self.does_it_cover(middle, ci_info):
                inner = middle.copy()
            else:
                outer = middle.copy()
            eps = np.abs(inner - outer)
        solution = (inner + outer) / 2
        return solution

    def aggregate_ci(self, L, U, S0, C, DF_RES, SES):

        low = np.min(C, axis=1)
        high = np.max(C, axis=1)

        ###### find an inside point: we need to find a point that is definitely in the confidence intervals

        range_len = 10
        l_bound = np.empty(self.d)
        u_bound = np.empty(self.d)

        for i in range(self.d):
            lci = L[:, i].copy()
            uci = U[:, i].copy()

            inf_ci = np.isinf(lci) + np.isinf(uci)

            center = C[~inf_ci, i].copy()
            ses = SES[~inf_ci, i].copy()

            no_inf_ci = sum(inf_ci)

            if no_inf_ci == len(lci) or no_inf_ci >= (1 - self.gamma_min) * len(lci):
                l_bound[i] = -np.Inf
                u_bound[i] = np.Inf
                continue

            lci = lci[~inf_ci]
            uci = uci[~inf_ci]
            ci_length = uci - lci

            df_res = DF_RES[~inf_ci]
            s0 = S0[~inf_ci]

            ci_Information = [lci, uci, ci_length, inf_ci, no_inf_ci, center, ses, s0, df_res]

            test_range = np.linspace(low[i], high[i], range_len)

            # does it cover gamma min?
            cover = np.array([False] * range_len)

            for j in range(range_len):
                cover[j] = self.does_it_cover(test_range[j], ci_Information)

            while not any(cover):
                range_len = range_len * 10
                test_range = np.linspace(low[i], high[i], range_len)
                cover = np.array([False] * range_len)
                for r in range(range_len):
                    cover[r] = self.does_it_cover(test_range[r], ci_Information)
                if range_len > 10 ** 3:
                    sys.exit("Find no inside point.")

            ##### Inner point
            inner = np.min(test_range[cover])
            outer = np.min(lci)

            ##### Generate new bounds
            ninner, nouter = self.find_bisection_bounds(inner, outer, ci_Information)
            l_bound[i] = self.bisection_gammamin_coverage(nouter, ninner, 10 ** (-7), ci_Information)

            outer = np.max(uci)

            ninner, nouter = self.find_bisection_bounds(ninner, outer, ci_Information)
            u_bound[i] = self.bisection_gammamin_coverage(nouter, ninner, 10 ** (-7), ci_Information)

        return l_bound, u_bound


class lasso_proj:
    def __init__(self, ci=True, ci_level=0.95, standardize=True, robust=False, family="gaussian", multicorr= True, verbose=True):

        """
        An implementation of LDPE projection method http://arxiv.org/abs/1110.2563
        in part based on R implementation lasso-proj.R by Ruben Dezeure
        ====================================================================================
        :param standardize: 'True' standardize the input data
        :param robust: 'True' use robust  sigma
        :param family: distribution of response variables
        ====================================================================================
        Attributes:
        pval: p-values for every parameter (individual tests)
        pval.corr:  multiple testing corrected p-values for every parameter
        betahat:    initial estimate by the scaled lasso of \beta^0
        bhat:       de-sparsified \beta^0 estimate used for p-value calculation
        sigmahat:   \sigma estimate coming from the scaled lasso
        ====================================================================================
        Author: Ziyan Zhu, Date: Oct 2nd,2019

        """
        self.ci = ci
        self.ci_level = ci_level
        self.standardize = standardize
        self.robust = robust
        self.family = family
        self.multicorr = multicorr
        self.verbose = verbose

    def cv_bestlambda(self, lambdas, x, K=10, lambdatuningfactor=1, model_select="ZZ", tuning=0.25):

        """
        ## Purpose:
        ## this function finds the optimal tuning parameter value for minimizing
        ## the K-fold cv error of the nodewise regressions.
        ## A second value of the tuning parameter, always bigger or equal to the
        ## former, is returned which is calculated by allowing the cv error to
        ## increase by the amount of
        ## 1 standard error (a similar concept as to what is done in cv.glmnet).
        """

        n, p = x.shape
        l = lambdas.shape[0]
        totalmse = np.zeros(shape=(1, l))
        # errmean = np.zeros(shape=(l,p))

        # perform Cross-validation by hands
        allsamp = np.tile(np.arange(1, 11), n)
        dataselects = np.random.choice(allsamp, n, replace=True)

        for c in range(p):
            X_j = np.copy(x)
            X_j = np.delete(X_j, c, axis=1)

            for i in range(1, (K + 1)):
                whichj = dataselects == i
                _, coefs, _ = lasso_path(X=X_j[~whichj, :], y=x[~whichj, c], alphas=lambdas,
                                         tol=0.001,max_iter =  1000)  # (n_features,n_alphas)
                predictions = np.dot(X_j[whichj, :], coefs)
                mse = np.mean((x[whichj, c:(c + 1)] - predictions) ** 2, axis=0)
                totalmse = totalmse + mse

        errmean = totalmse / (K * p)
        pos_min = np.where(errmean == np.min(errmean))[0]
        bestlam = np.min(lambdas[pos_min]) * lambdatuningfactor

        if model_select == "ZnZ" or model_select == "znz":  # improve the lambda with Z&Z procedures
            # eta = np.sqrt(2 * np.log10(p))  # target bound for bias factor

            noise = np.zeros(shape=(p, l))
            for c in range(p):
                X_j = np.copy(x)
                X_j = np.delete(X_j, c, axis=1)
                _, coefs, _ = lasso_path(X=X_j, y=x[:, c], alphas=lambdas)
                # coefs (n_features,n_alphas)

                zj = x[:, c:(c + 1)] - np.dot(X_j, coefs)
                # zj (n_samples,n_alphas)

                # eta = np.dot(X_j.T,zj)/np.linalg.norm(zj,axis=0)
                noisej = np.linalg.norm(zj, axis=0) / np.dot(x[:, c:(c + 1)].T, zj)
                # (1,n_alphas)
                noise[c, :] = noisej.T

            noise = np.mean(noise, axis=0)

            pos_min = np.where(lambdas == np.min(lambdas) and lambdas > 0)[0][0]
            bestlam = lambdas[pos_min]
            # the smallest non-zero penalty is initially set as the bestlambda
            # or should we use the lambda_min selected by smallest mse?
            noise_opt = noise[pos_min]

            if (noise < (1 + tuning) * noise_opt).any():
                bestlam = np.min(lambdas[np.where(noise < (1 + tuning) * noise_opt)[0]])

            # the lambdas were already sorted from big to small
            if np.max(np.where(noise < (1 + tuning) * noise_opt)[0]) < l - 1:
                # there is an interval of potential small lambda values that give close to the 25% inflation

                pos_min = np.max(np.where(noise < (1 + tuning) * noise_opt)[0])

                newlams = np.linspace(lambdas[pos_min], lambdas[pos_min + 1], 100)
                newlams = -np.sort(-newlams)  # just in case the lambdas are not sorted decreasing
                noise = np.zeros(shape=(p, l))
                for c in range(p):
                    X_j = np.copy(x)
                    X_j = np.delete(X_j, c, axis=1)
                    _, coefs, _ = lasso_path(X=X_j, y=x[:, c], alphas=newlams)
                    # coefs (n_features,n_alphas)

                    zj = x[:, c:(c + 1)] - np.dot(X_j, coefs)
                    # zj (n_samples,n_alphas)

                    # eta = np.dot(X_j.T,zj)/np.linalg.norm(zj,axis=0)
                    noisej = np.linalg.norm(zj, axis=0) / np.dot(x[:, c:(c + 1)].T, zj)
                    # (1,n_alphas)
                    noise[c, :] = noisej.T
                noise = np.mean(noise, axis=0)
                if (noise < (1 + tuning) * noise_opt).any():
                    bestlam = np.min(lambdas[np.where(noise < (1 + tuning) * noise_opt)[0]])

        return bestlam

    def sandwich_robust_se(self, x, y, betainit, Z):
        """
        ## Purpose:
        ## an implementation of the calculation of the robust standard error
        ## based on the sandwich variance estimator from
        ## http://arxiv.org/abs/1503.06426

        :param x: the design matrix
        :param y: the response vector
        :param betainit: the initial estimate
        :param Z: the residuals of the nodewis regressions
        :return: sigmahatZ
        """
        n, p = x.shape

        # check if normalization is fullfiled

        if ~np.allclose(np.ones(p), np.sum(Z * x / n, axis=0)):
            dz = np.dot(Z.T, x)
            dz = np.diag(dz)
            scaleZ = dz / n
            Z = np.true_divide(Z, scaleZ)


        if len(betainit) > p:
            x = np.column_stack(np.ones(n), x)

        eps_tmp = y - np.dot(x, betainit)

        ## force esp_tmp to have mean 0 as if we fit with intercept
        eps_tmp = eps_tmp - np.mean(eps_tmp)
        stats = np.dot(eps_tmp, Z) / n
        eps_tmp = eps_tmp.reshape((n, 1))
        save = eps_tmp * Z
        sweep = save - stats
        sigmahatZ = np.sqrt(np.sum(sweep ** 2, axis=0)) / n

        return sigmahatZ

    def nodewiselasso(self, X, y, lambdaseq="quantile", lambdatuningfactor=1):
        if self.standardize:
            if np.all(np.mean(X, axis=0)==0):
                pass
            else:
                sds = np.std(X, axis=0, ddof=1)
                # center the columns to get rid of the intercept#
                X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0, ddof=1))

                if self.family == "binomial":
                    X, y = switch_binomial(X, y)

                x = (X - np.mean(X, axis=0))
                y = (y - np.mean(y, axis=0))

        # Find a good score vector zj to approximate the weight vector wj
        n, p = X.shape

        # According to Friedman, Hastie & Tibshirani (2010) 'strategy is to select a minimum value lambda_min = epsilon * lambda_max, and construct a sequence of K values of lambda decreasing from lambda_max to lambda_min on the log scale.

        eps = 0.001
        K = 100
        lams = np.array([])

        for c in range(p):
            temp = np.copy(x)
            temp = np.delete(temp, c, axis=1)
            lambda_path,_,_=lasso_path(temp,x[:,c])
            #max_lambda = np.max(np.abs(np.sum(np.dot(temp.T, x[:, c])))) / n
            # the lambda that allows the greatest penalty, and all beta entries are zero
            #lambda_path = np.round(np.exp(np.linspace(math.log10(max_lambda * eps), math.log10(max_lambda), K)),decimals=100)
            lams = np.append(lams, lambda_path)  # the complete set of lambda values

        # Get a sequence of lambda values over all regressions for selection
        if lambdaseq == "quantile":  # Equidistant quantiles
            seq = np.linspace(0, 1, K, dtype=float)
            lams = -np.sort(-lams)
            lambdas = np.quantile(lams, seq)

        if lambdaseq == "linear":  # A linear interpolation
            lambdas = np.linspace(np.min(lams), np.max(lams), K)

        # Find the desired lambda:
        # Use 10-fold cv to find the best lambda with the minimized error.

        bestlambda = self.cv_bestlambda(lambdas, x, K=3, lambdatuningfactor=lambdatuningfactor)

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

    def fit(self, X, y):

        ###### Prepare the data
        n, p = X.shape

        if self.standardize:
            sds = np.std(X, axis=0, ddof=1)
        else:
            sds = np.ones(p)

        # center the columns to get rid of the intercept
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0, ddof=1))

        if self.family == "binomial":
            X, y = switch_binomial(X, y)

        x = (X - np.mean(X, axis=0))
        y = (y - np.mean(y, axis=0))

        ###### Calculate score matrix Z using nodewise lasso

        if self.verbose:
            print("Calculating Z ...")

        # Z is the residuals of the nodewise regressions under 10-folds cv
        scaleZ, Z = self.nodewiselasso(x, y)

        #scaleZ = np.loadtxt('/Users/alexazhu/Desktop/XY6_R/scaleZ.csv',delimiter=",")
        #Z = np.loadtxt('/Users/alexazhu/Desktop/XY6_R/Z.csv', delimiter=",")
        ##### Projection estimator and bias

        bproj = np.dot(Z.T, y) / n

        # get initial estimators using Lasso with 10-folds cross-validation

        model = LassoCV(cv=10, n_jobs=2).fit(x, y)
        hbeta = model.coef_

        hy = model.predict(x)
        hsigma = np.sqrt(np.sum(np.square(y - hy)) / (n - np.sum(hbeta != 0)))
        #np.savetxt("/Users/alexazhu/Desktop/Python_hsigma.csv", hsigma)

        #hbeta = np.loadtxt('/Users/alexazhu/Desktop/XY6_R/betalasso.csv', delimiter=",")

        #hsigma = 1.076852

        if self.verbose:
            print("Calculating bias ... ")

        #### Subtract bias ####
        bias = np.zeros(p)
        for j in range(p):
            temp = np.copy(x)
            temp = np.delete(temp, j, axis=1)
            betatemp = np.copy(hbeta)
            betatemp = np.delete(betatemp, j)
            bias[j] = np.dot(np.dot(Z[:, j].T, temp), betatemp) / n

        bproj = bproj - bias

        #np.savetxt("/Users/alexazhu/Desktop/Python_bproj.csv",bproj)
        ######## calculate p-values #######

        if self.robust:
            sigmahatZ = self.sandwich_robust_se(x, y, hbeta, Z)
            scaleb = (1 / sigmahatZ)
        else:
            scaleb = n / (hsigma * np.sqrt(np.sum(Z ** 2, axis=0)))

        self.se = 1 / scaleb
        bprojrescaled = bproj * scaleb
        self.bhat = bproj / sds

        pvals = 2  * norm.sf(np.abs(bprojrescaled))
        self.pvals = pvals

        #np.savetxt("/Users/alexazhu/Desktop/Python_pval.csv", pvals)

        ######## adjust p-values for multiple testing ########
        cov2 = np.dot(Z.T, Z)

        ######## calculate confidence interval
        if self.ci:
            self.uci = self.bhat + norm.ppf(1 - ci_level / 2) * self.se
            self.lci = self.bhat - norm.ppf(1 - ci_level / 2) * self.se


class ridge_proj:
    def __init__(self, standardize=True, ridge_unprojected=False, family="gaussian"):
        self.family = family
        self.standardize = standardize
        self.ridge_unprojected = ridge_unprojected

    def tsvd(self, A):
        n, p = A.shape
        tol = min(n, p) * np.finfo(np.float64).eps
        tsvd = TruncatedSVD(n_components=min(n, p), tol=tol)
        tsvd.fit(A)

        Sval = tsvd.singular_values_

        rankA = np.count_nonzero(Sval >= (tol * Sval[0]))

        u, s, vh = np.linalg.svd(A, full_matrices=True)
        return u[:, :rankA], s[:rankA].reshape((rankA, 1)), vh.T[:, :rankA]

    def fit(self, X, y, lam=1):
        n, p = X.shape

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

        ## SVD
        U, S, V = self.tsvd(X)

        Px = np.dot(V, V.T)
        Px_offdiag = Px.copy()
        np.fill_diagonal(Px_offdiag, 0)
        ## Use svd for getting the inverse for the ridge problem

        Omiga = np.dot(V, (S / (S ** 2 + lam)) * U.T)

        ## Ruben Note: here the S^2/n 1/n factor has moved out, the value of lambda
        ## used is 1/n! See also comparing to my version

        cov2 = np.dot(Omiga, Omiga.T)
        diag_cov2 = np.diagonal(cov2)

        # get initial estimators with LassoCV
        model = LassoCV(cv=10, n_jobs=-1).fit(X, y)  # use 3 CPUs
        betainit = model.coef_
        hy = model.predict(X)
        hsigma = np.sqrt(np.sum(np.square(y - hy)) / (n - np.sum(betainit != 0)))

        hsigma2 = hsigma ** 2

        ## bias correction
        biascorr = np.dot(Px_offdiag.T, betainit)

        ## ridge estimator
        hbeta = np.dot(Omiga, y)

        hbetacorr = hbeta - biascorr

        if self.ridge_unprojected:  ## bring it back to the original scale
            hbetacorr = hbetacorr / np.diagonal(Px)

        ## Ruben Note: a_n = 1 / scale.vec, there is no factor sqrt(n) because this
        ## falls away with the way diag.cov2 is calculated see paper
        scale_vec = np.sqrt(hsigma2 * diag_cov2)

        if self.ridge_unprojected:
            scale_vec = scale_vec / abs(np.diagonal(Px))

        self.bhat = hbetacorr / sds

        hbetast = hbetacorr / scale_vec

        Delta = (1 / scale_vec) * np.max(abs(Px_offdiag), axis=0) * ((np.log10(p) / n) ** 0.45)

        if (self.ridge_unprojected):  ## 2
            Delta = Delta / abs(np.diag(Px))

        hgamma = abs(hbetast)

        ## Individual p-values
        temp = 2 * norm.sf(np.abs(hgamma - Delta))

        self.pvals = np.minimum(temp, 1)

        ## Multiple testing corrected p-values ##

        self.pvals_corr = self.pval_adjust_WY(cov2, self.pvals)


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

    def inverseLinftyOneRow(self, sigma, i, mu, maxiter=50, threshold=1e-2):
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
                beta[j] = (np.sign(v) * max(0, abs(v) - mu)) / sigma[j, j]  # Soft thresholding
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

    def inverseLinfty(self, sigma, n, resol=1.5, mu=None, maxiter=50, threshold=1e-2):
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

    def binarySearch(self, X, y, select_num=50):
        model = Lasso()
        betaM = np.zeros([X.shape[1]])
        # max(abs(crossprod(xx, yy / sqrt(sum(yy ^ 2)) / sqrt(n))))
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

    def fit(self, X, y, lmbd=None, intercept=False):
        """
        :param X: design matrix
        :param y: response
        :param intercept:
        :param lmbd: Lasso regularization parameter (if null, fixed by sqrt lasso)

        :return:
            p-values, confidence intervals
        """

        print("preparing")
        n, p = X.shape
        pp = p
        col_norm = 1.0 / np.sqrt((1.0 / n) * np.diag(np.dot(X.T, X)))
        X = np.dot(X, np.diag(col_norm))

        if lmbd == None:
            lmbd = np.sqrt(norm.ppf(1 - (0.1 / p)) / n)

        # Objective : sqrt(RSS/n) +lambda *penalty

        # sigma_hat_diag = np.zeros(p)
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
            Xb = np.concatenate(np.ones(n), X)
            col_norm = np.concatenate(1, col_norm)
            p = p + 1
        else:
            Xb = X

        sigma_hat = (1.0 / n) * np.dot(Xb.T, Xb)

        # X2 = np.zeros(p)
        # for i in range(p):
        # X2[i] = np.dot(X[:, i], X[:, i].T)

        if n >= 2 * p:
            tmp = np.linalg.eig(sigma_hat)
            tmp = np.min(tmp) / np.max(tmp)
        else:
            tmp = 0

        if n >= 2 * p and tmp >= 1e-4:
            M = np.linalg.inv(sigma_hat)
        else:
            M = self.inverseLinfty(sigma=sigma_hat, n=n, resol=1.3, mu=None, maxiter=50, threshold=1e-2)

        A = np.dot(np.dot(M, sigma_hat), M.T)

        print('fitting')
        # self.model.fit(X, y)
        # beta = self.model.coef_
        # hbeta =self.binarySearch(X, y)
        thetamodel = LassoCV(fit_intercept=intercept)
        thetamodel.fit(X, y)
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


def hd(method="multi-split", X=None, y=None, family="gaussian",
       ci=False, ci_level=0.95, B=100, fraction=0.5, repeat_max=20, manual_lam=True,
       aggr_ci=True, return_nonaggr=False,
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
        est = multi_split(ci=ci, ci_level=ci_level, B=B, fraction=fraction, manual_lam=manual_lam, aggr_ci=aggr_ci,
                          return_nonaggr=return_nonaggr, model_selector=model_selector, repeat_max=repeat_max)
        est.fit(X=X, y=y)
        return est

    if method == "lasso-proj" or method == "lassoproj":
        est = lasso_proj(ci=ci, ci_level=ci_level, standardize=standardize, robust=robust, family=family)
        est.fit(X=X, y=y)
        return est

    if method == "ridge-proj" or method == "ridgeproj":
        est = ridge_proj(standardize=standardize, ridge_unprojected=ridge_unprojected, family=family)
        est.fit(X=X, y=y)
        return est

    if method == "debiasedLasso" or method == "debiased":
        print("No corrected p-values for debiasedLasso.")
        est = debiasedLasso()
        est.fit(X, y)

        return est


def manhattanplot(position, pvalues):
    # Set up the matplotlib figure
    if position==None:
        position = np.arange(1,len(pvalues))
    sns.set_style("whitegrid")
    sns.distplot(pvalues,position,color="m")
    sns.plt.show()
