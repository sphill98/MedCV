import mat73
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import os


# Load .mat file
def loadMAT(path, varname=None):
    try:
        f = mat73.loadmat(path)   # only_include=self.varname[i]) 
    except: 
        f = io.loadmat(path)
    if varname is None:
        return f
    else:
        return f[varname]
    

if __name__ == '__main__':
    matFile = loadMAT()
    
