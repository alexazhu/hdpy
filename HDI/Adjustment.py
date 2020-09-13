import numpy as np
import math
import sys


def control_FDR(pvals_corr, q):
    d = pvals_corr.shape[0]
    ## FDR control at q level
    pval_order = np.sort(pvals_corr)  # ascending
    index = np.arange(1, d + 1, 1, dtype=int)
    h = np.max(np.where(pvals_corr <= index * q / np.sum(1 / index))[0])
    active_FDR = np.where(pvals_corr <= pval_order[h])[0]
    return active_FDR


def control_FWER(pvals_corr, level):
    active_FWER = np.where(pvals_corr <= level)[0]
    return active_FWER

