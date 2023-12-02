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
    
def padding(data, arraySize):
    # Make the array the same size as arraySize
    # Write your code here
    z, y, x = data.shape
    print((z, y, x))
    pad_z = max(0, arraySize - z)
    pad_y = max(0, arraySize - y)
    pad_x = max(0, arraySize - x)

    padded_data = np.pad(data, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
    return padded_data
        

if __name__ == '__main__':
    matFile = loadMAT('/Users/siyeol/2023-2/software_college_project/train/vertebrae004.mat')

    print(matFile['rawTemp'].shape)
    temp = np.ascontiguousarray(matFile['rawTemp'])

    padded_temp = padding(temp, arraySize=256)

    print(padded_temp.shape)

    padded_temp = padded_temp.transpose(1, 2, 0)

    print(padded_temp.shape)

    plt.imshow(padded_temp[30])
    plt.show()
