# hdpy

## Introduction

How do we do inference?

Confidence intervals and p-values provide some uncertainty accessment of the parameter estimatioins.


The hdpy package is a Python module for statistical inference with linear regression models on high-dimensional data.  
* Implementation of four high-dimensional regression methods: Multiple splitting Method, Lasso Projection method, Ridge Projection method, debiased Lasso method.  
* Produce p-values adjusted for multiple testing.
* Aggregate confidence intervals for non-zero coefficients.
* Visualize p-values for variable screening


## File Structure

* HDI/ four individual implementations of high-dimensional inference methods.
* data/ high dimensional data simulator with experiment dataset in the example.
* hdpy.py main entry point of using the hdpy to work with your own data


## Installation
Numpy, Sklearn, Statsmodels, seaborn package are required on your current system.


## Example

### Generate and Save the Simualtion Data
```
import numpy as np
from Data_Simulate import simulation

sim = simulation(n=50, p=500, snr=0.9, type='Toeplitz',seed=1)
betas, active_index = sim.get_coefs(s0=8, random_coef=False, b=1,save=True,path="./",filename="coefs")
X,Y= sim.get_data(verbose=False,save=True,
                 path="./",filename="data")

```

### Fit the Data
```
import hdpy
est = hdpy.hd(method = "multi-split", X=X,y=Y,...)
```

### Get p-values / confidence intervals
```
pvals = est.pvals\
pvals_corr = est.pvals_corr\
#lci = est.lci\
#uci = est.uci
```
### Data Visualization






