import mat73
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import os


# Load .mat file
def loadMAT(path, varname=None):
    try:
        f = mat73.loadmat(path)  # only_include=self.varname[i])
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

class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        label, input_ct = data[0], data[1]

        label = label.astype(np.float32)
        input_ct = input_ct.astype(np.float32)

        if len(input_ct.shape) == 2:
            y, x = input_ct.shape
            crop_height, crop_width = self.shape

            start_y = np.random.randint(0, max(1, y - crop_height + 1))
            start_x = np.random.randint(0, max(1, x - crop_width + 1))

            cropped_label = label[start_y:start_y + crop_height, start_x:start_x + crop_width].astype(np.float32)
            cropped_input_ct = input_ct[start_y:start_y + crop_height, start_x:start_x + crop_width].astype(np.float32)
        elif len(input_ct.shape) == 3:
            z, y, x = input_ct.shape
            crop_depth, crop_height, crop_width = self.shape

            if crop_depth >= z or crop_height >= y or crop_width >= x:
                raise ValueError("Crop dimensions are larger than the input dimensions.")

            start_z = np.random.randint(0, max(1, z - crop_depth + 1))
            start_y = np.random.randint(0, max(1, y - crop_height + 1))
            start_x = np.random.randint(0, max(1, x - crop_width + 1))

            cropped_label = label[start_z:start_z + crop_depth, start_y:start_y + crop_height, start_x:start_x + crop_width].astype(np.float32)
            cropped_input_ct = input_ct[start_z:start_z + crop_depth, start_y:start_y + crop_height, start_x:start_x + crop_width].astype(np.float32)
        else:
            raise ValueError("Input data not 2D/3D")

        cropped_data = {'label': cropped_label, 'input': cropped_input_ct}
        return cropped_data



if __name__ == '__main__':
    matFile = loadMAT('vertebrae004.mat')

    print(matFile['rawTemp'].shape)
    temp = np.ascontiguousarray(matFile['rawTemp'])

    print(temp.shape)

    padded_temp = padding(temp, arraySize=256)

    print(padded_temp.shape)

    padded_temp = padded_temp.transpose(1, 2, 0)

    print(padded_temp.shape)

    random_crop = RandomCrop(shape=(128, 128))
    cropped_temp = random_crop(temp)

    print(cropped_temp['label'].shape)
    print(cropped_temp['input'].shape)
