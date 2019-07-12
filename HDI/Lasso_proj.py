__author__ = 'Ziyan Zhu'

import numpy as np
import math
from scipy.stats import norm
from statsmodels.distributions import ECDF
from sklearn.linear_model import Lasso,LassoCV,lasso_path,LassoLars, LogisticRegressionCV


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

class lassoproj:
    ## An implementation of the Z&Z projection method http://arxiv.org/abs/1110.2563
    ## Arguments:
    ## ----------------------------------------------------------------------
    ## Return values:
    ## pval: p-values for every parameter (individual tests)
    ## pval.corr:  multiple testing corrected p-values for every parameter
    ## betahat:    initial estimate by the scaled lasso of \beta^0
    ## bhat:       de-sparsified \beta^0 estimate used for p-value calculation
    ## sigmahat:   \sigma estimate coming from the scaled lasso
    ## ----------------------------------------------------------------------
    ## Author: Ziyan Zhu, Haohan Wang Date: 18 Apr 2019 (initial version),
    ## in part based on an implementation of the lasso projection method
    ## lasso-proj.R by Ruben Dezeure

    def __init__(self, standardize=True,robust=True,family = "gaussian",verbose = True):
        self.standardize = standardize
        self.robust = robust
        self.family = family
        self.verbose = verbose

    def pval_adjust_WY(self, cov, pvals, N=10000):
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
        pcorr = ecdf(pvals)
        return pcorr

    def cv_bestlambda(self,lambdas,x,K=10,lambdatuningfactor=1,model_select="ZZ",tuning = 0.25):

        ## Purpose:
        ## this function finds the optimal tuning parameter value for minimizing
        ## the K-fold cv error of the nodewise regressions.
        ## A second value of the tuning parameter, always bigger or equal to the
        ## former, is returned which is calculated by allowing the cv error to
        ## increase by the amount of
        ## 1 standard error (a similar concept as to what is done in cv.glmnet).

        n,p=x.shape
        l=lambdas.shape[0]
        totalmse = np.zeros(shape=(1, l))
        # errmean = np.zeros(shape=(l,p))

        # perform Cross-validation by hands
        allsamp = np.tile(np.arange(1,11),n)
        dataselects = np.random.choice(allsamp,n,replace=True)

        for c in range(p):
            X_j = np.copy(x)
            X_j = np.delete(X_j, c, axis=1)

            for i in range(1,(K+1)):
                whichj = dataselects==i
                _,coefs,_= lasso_path(X=X_j[~whichj, :], y=x[~whichj,c], alphas=lambdas,tol=0.001) # (n_features,n_alphas)
                predictions = np.dot(X_j[whichj, :],coefs)
                mse = np.mean((x[whichj,c:(c+1)]-predictions)**2,axis=0)
                totalmse = totalmse + mse

        errmean = totalmse / (K * p)
        pos_min = np.where(errmean == np.min(errmean))[0]
        bestlam =  np.min(lambdas[pos_min]) * lambdatuningfactor


        if model_select =="ZnZ" or model_select =="znz": #improve the lambda with Z&Z procedures
            #eta = np.sqrt(2 * np.log10(p))  # target bound for bias factor

            noise = np.zeros(shape=(p,l))
            for c in range(p):
                X_j = np.copy(x)
                X_j = np.delete(X_j, c, axis=1)
                _, coefs, _ = lasso_path(X=X_j, y=x[:, c], alphas=lambdas)
                 # coefs (n_features,n_alphas)

                zj = x[:,c:(c+1)]-np.dot(X_j,coefs)
                 # zj (n_samples,n_alphas)

                #eta = np.dot(X_j.T,zj)/np.linalg.norm(zj,axis=0)
                noisej = np.linalg.norm(zj,axis=0)/np.dot(x[:,c:(c+1)].T,zj)
                    # (1,n_alphas)
                noise[c,:] = noisej.T

            noise = np.mean(noise,axis=0)

            pos_min = np.where(lambdas == np.min(lambdas) and lambdas > 0)[0][0]
            bestlam = lambdas[pos_min]
            # the smallest non-zero penalty is initially set as the bestlambda
            # or should we use the lambda_min selected by smallest mse?
            noise_opt = noise[pos_min]

            if (noise < (1+tuning)*noise_opt).any():
                bestlam = np.min(lambdas[np.where(noise<(1+tuning)*noise_opt)[0]])

            # the lambdas were already sorted from big to small
            if np.max(np.where(noise<(1+tuning)*noise_opt)[0]) < l-1:
                # there is an interval of potential small lambda values that give close to the 25% inflation

                pos_min = np.max(np.where(noise<(1+tuning)*noise_opt)[0])

                newlams = np.linspace(lambdas[pos_min],lambdas[pos_min+1],100)
                newlams = -np.sort(-newlams) #just in case the lambdas are not sorted decreasing
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

    def nodewiselasso(self,x,y,lambdaseq="quantile",lambdatuningfactor=1):


        if self.standardize:
            if np.mean(x,axis=0).all()==0:
                pass
            else:
                sds = np.std(x, axis=0, ddof=1)
                # center the columns to get rid of the intercept#
                x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0, ddof=1))

                if self.family == "binomial":
                    x, y = switch_binomial(x, y)

                x = (x - np.mean(x, axis=0))
                y = (y - np.mean(y, axis=0))

        #Find a good score vector zj to approximate the weight vector wj
        n,p = x.shape

        #According to Friedman, Hastie & Tibshirani (2010) 'strategy is to select a minimum value lambda_min = epsilon * lambda_max, and construct a sequence of K values of lambda decreasing from lambda_max to lambda_min on the log scale.

        eps = 0.001
        K = 100
        lams = np.array([])

        for c in range(p):
            temp = np.copy(x)
            temp = np.delete(temp, c, axis=1)
            max_lambda = np.max(np.abs(np.sum(np.dot(temp.T, x[:, c])))) / n
             # the lambda that allows the greatest penalty, and all beta entries are zero
            lambda_path = np.round(np.exp(np.linspace(math.log10(max_lambda*eps), math.log10(max_lambda), K)),
                                   decimals=100)
            lams = np.append(lams,lambda_path) # the complete set of lambda values

        # Get a sequence of lambda values over all regressions for selection
        if lambdaseq=="quantile": # Equidistant quantiles
            seq = np.linspace(0, 1, K,dtype=float)
            lams = -np.sort(-lams)
            lambdas = np.quantile(lams,seq)

        if lambdaseq=="linear": # A linear interpolation
            lambdas = np.linspace(np.min(lams),np.max(lams),K)


        ## Find the desired lambda:
        #     Use 10-fold cv to find the best lambda with the minimized error.

        bestlambda=self.cv_bestlambda(lambdas,x,K=10,lambdatuningfactor=lambdatuningfactor)

        Z = np.zeros(shape=(n, p))

        getZ = Lasso(alpha=bestlambda,max_iter=100)

        for i in range(p):
            temp = np.copy(x)
            temp = np.delete(temp, i, axis=1)
            getZ = getZ.fit(X=temp, y=x[:, i])
            prediction = getZ.predict(X=temp)
            Z[:, i] = x[:, i] - prediction

        ## rescale Z such that t(Zj) Xj/n = 1 for all j
        scaleZ = np.diag(np.dot(Z.T, x))/n
        Z = np.true_divide(Z, scaleZ)

        return scaleZ,Z

    def sandwich_robust_se(self,x,y,betainit,Z):
        ## Purpose:
        ## an implementation of the calculation of the robust standard error
        ## based on the sandwich variance estimator from
        ## http://arxiv.org/abs/1503.06426
        ## ----------------------------------------------------------------------
        ## Arguments:
        ## x: the design matrix
        ## y: the response vector
        ## betainit: the initial estimate
        ## Z:       the residuals of the nodewis regressions
        ## ----------------------------------------------------------------------
        n,p = x.shape

        # check if normalization is fullfiled

        if ~np.allclose(np.ones(p), np.sum(Z*x/ n, axis=0)):
            dz = np.dot(Z.T, x)
            dz = np.diag(dz)
            scaleZ = dz / n
            Z = np.true_divide(Z, scaleZ)

        if len(betainit) > p:
            x = np.column_stack(np.ones(n), x)

        eps_tmp = y - np.dot(x,betainit)

        ## force esp_tmp to have mean 0 as if we fit with intercept
        eps_tmp = eps_tmp - np.mean(eps_tmp)
        stats = np.dot(eps_tmp, Z) / n
        eps_tmp = eps_tmp.reshape((n, 1))
        save = eps_tmp * Z
        sweep = save - stats
        sigmahatZ = np.sqrt(np.sum(sweep ** 2, axis=0)) / n

        return sigmahatZ

    def fit(self, x, y):

        ##### Prepare the data. #######

        n,p = x.shape

        if self.standardize:
            sds = np.std(x, axis=0, ddof=1)
        else:
            sds = np.ones(p)

        # center the columns to get rid of the intercept#
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0, ddof=1))

        if self.family=="binomial":
            x,y = switch_binomial(x,y)

        x = (x - np.mean(x, axis=0))
        y = (y - np.mean(y, axis=0))


        ##### Calculate score vectors Z using nodewise lasso #####

        if self.verbose:
            print("Calculating Z ...")

        # Z is the residuals of the nodewise regressions under 10-folds cv
        scaleZ,Z = self.nodewiselasso(x,y)

        ##### Projection estimator and bias #####

        bproj = np.dot(Z.T, y) / n

        # get initial estimators using Lasso with 10-folds cross-validation

        model = LassoCV(cv=10,n_jobs=-1).fit(x,y) #use all CPUs
        hbeta = model.coef_
        hy = model.predict(x)
        hsigma = np.sqrt(np.sum(np.square(y - hy)) / (n - np.sum(hbeta != 0)))

        if self.verbose:
            print("Calculating bias ... ")

        #### Subtract bias ####
        bias = np.zeros(p)
        for j in range(p):
            temp = np.copy(x)
            temp = np.delete(temp, j, axis=1)
            betatemp = np.copy(hbeta)
            betatemp = np.delete(betatemp, j)
            bias[j] = np.dot(np.dot(Z[:, j].T, temp),betatemp)/n
        bproj = bproj - bias

        ######## calculate p-values #######

        if self.robust == True:
            sigmahatZ= self.sandwich_robust_se(x,y,hbeta,Z)
            scaleb = 1 / sigmahatZ
        else:
            scaleb = n / (hsigma * np.sqrt(np.sum(Z ** 2, axis=0)))

        self.se = 1 / scaleb
        bprojrescaled = bproj * scaleb
        self.bhat = bproj / sds

        pvals =  2 * norm.sf(np.abs(bprojrescaled))
        self.pvals = pvals

        ######## adjust p-values for multiple testing ########
        cov2 = np.dot(Z.T, Z)
        pvals_corr = self.pval_adjust_WY(cov2, pvals)

        self.pvals_corr = pvals_corr

    def ci(self,ci_level):
        ######## calculate confidence interval
        self.ci = [self.bhat - norm.ppf(1-ci_level/2) *self.se, self.bhat + norm.ppf(1-ci_level/2)*self.se]





