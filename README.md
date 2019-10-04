# hdpy package
A python module for statistical inference with regularized linear regression model on high-dimensional data.  
* High-dimensional data with p>>n  
* P-values adjusted for multiple testing.
* Confidence intervals.

## installation
import hdpy

## fitting
data = np.loadtxt(file,delimiter=',')\
X = data[:,1:]\
Y = data[:,0:1]\
Y = Y.reshape((100,))\
est = hdpy.hd(method = "multi-split", X=X,y=Y,...)

## get p-values / confidence intervals
pvals = est.pvals\
pvals_corr = est.pvals_corr\
#lci = est.lci\
#uci = est.uci

