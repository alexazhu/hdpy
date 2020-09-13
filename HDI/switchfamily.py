from sklearn.linear_model import LogisticRegressionCV
import numpy as np

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


