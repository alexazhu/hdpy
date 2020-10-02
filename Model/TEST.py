import numpy as np
from Data_Simulate import simulation
from sklearn.linear_model import LassoCV,LassoLars,Lasso
from sklearn import preprocessing
import hdpy
import math

path = '/Users/alexazhu/Desktop/Data1002/'

sim = simulation(n=50, p=100, snr=0.9, type="Toeplitz",seed=1)
betas, active = sim.get_coefs(s0=None, random_coef=False, b=1, unif_up=None, unif_lw=None,save=True, path=path,filename="data")

X,y = sim.get_data(save=True,path=path,filename="data")
#print(hdpy.hd(method="multi-split",X=X,y=y,ci=False,B=100))
#print(hdpy.hd(method="lasso-proj",X=X,y=y,ci=False,B=100))
print(hdpy.hd(method="ridge-proj",X=X,y=y,ci=False,B=100))
print(hdpy.hd(method="debiased-lasso",X=X,y=y,ci=False,B=100))

print("The original active set is: ")
print(active)
