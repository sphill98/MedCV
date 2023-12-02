from utility import *
from loadMAT import loadMAT


# Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, plane='axial', filename=None, varname=['segTemp', 'rawTemp'], arraySize=256):
        self.data_path = data_path
        self.transform = transform
        self.plane = plane
        self.filename = filename  # [2,N] shape strin list, [ref, target]
        self.varname = varname  # 2 string [ref, target]
        self.arraySize = arraySize
        self.data_array = self.preprocessing()
        self.ToTensor = ToTensor()
        print('Dataset:DATA SIZE = ' + str(self.data_array.shape))  # [:, N, Z, Y, X]

    def __len__(self):
        return self.data_array.shape[1]

    def __getitem__(self, index):
        data1 = self.data_array[0, index]  # Label  [Z, Y, X]
        data2 = self.data_array[1, index]  # Input CT image
        data = {'label': data1, 'input': data2}

        if self.transform is not None:
            data = self.transform(data)

        data = self.ToTensor(data)

        return data

    def preprocessing(self):
        for i in range(len(self.filename)):
            fileID = os.path.join(self.data_path, self.filename[i][0])
            f = loadMAT(fileID) 
            for j in range(2):
                temp = f[self.varname[j]]
                temp = np.ascontiguousarray(temp)
                # Padding 
                temp = self.padding(temp)
                if self.plane == 'axial':
                    temp = temp.transpose((0, 1, 2))[None, :, None, :]
                elif self.plane == 'sagittal':
                    temp = temp.transpose((1, 2, 0))[None, None, :, :]
                elif self.plane == 'coronal':
                    temp = temp.transpose((2, 1, 0))[None, None, :, :]
                else:
                    print('ERROR: plane is not defined')
                    exit()
                datatemp = temp if j==0 else np.concatenate((datatemp, temp), axis=0)
            if i==0 and j==0:
                self.minmax = [np.min(datatemp), np.max(datatemp)]
            data_array = datatemp if i==0 else np.concatenate((data_array, datatemp), axis=1)
        data_array = self.minmaxNorm(data_array)

        return data_array

    def minmaxNorm(self, data):
        return (data - self.minmax[0]) / (self.minmax[1] - self.minmax[0])
    
    def minmaxDenorm(self, data):
        return data * (self.minmax[1] - self.minmax[0]) + self.minmax[0]
    
    def inversePlaneTransform(self, data):
        if self.plane == 'axial':
            data = data[0]
        elif self.plane == 'sagittal':
            data = data[0].transpose((1, 2, 0))
        elif self.plane == 'coronal':
            data = data[0].transpose((2, 1, 0))
        else:
            print('ERROR: plane is not defined')
            exit()
        return data
    
    def padding(self, data):
        # Make the array the same size as arraySize
        pad_values = [(0, max(0, self.arraySize[0] - data.shape[0])),
                      (0, max(0, self.arraySize[1] - data.shape[1])),
                      (0, max(0, self.arraySize[2] - data.shape[2]))]
        data = np.pad(data, pad_values, mode='constant', constant_values=0)
        return data


class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.astype(np.float32)
            data[key] = torch.from_numpy(value)
        return data


class RandomRot90(object):
    def __call__(self, data):
        for key, value in data.items():
            plane = np.random.choice(['axial', 'sagittal', 'coronal'])
            if plane == 'axial':
                value = np.rot90(value, k=1, axes=(1, 2))
            elif plane == 'sagittal':
                value = np.rot90(value, k=1, axes=(0, 2))
            elif plane == 'coronal':
                value = np.rot90(value, k=1, axes=(0, 1))
            data[key] = value
        return data


class RandomFlip(object):
    def __call__(self, data):
        for key, value in data.items():
            plane = np.random.choice(['axial', 'sagittal', 'coronal'])
            if plane == 'axial':
                if np.random.rand() > 0.5:
                    value = np.flip(value, axis=1)
                if np.random.rand() > 0.5:
                    value = np.flip(value, axis=2)
            elif plane == 'sagittal':
                if np.random.rand() > 0.5:
                    value = np.flip(value, axis=0)
                if np.random.rand() > 0.5:
                    value = np.flip(value, axis=2)
            elif plane == 'coronal':
                if np.random.rand() > 0.5:
                    value = np.flip(value, axis=0)
                if np.random.rand() > 0.5:
                    value = np.flip(value, axis=1)
            data[key] = value
        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        z, y, x = self.shape
        for key, value in data.items():
            _, _, dz, dy, dx = value.shape
            if dz > z and dy > y and dx > x:
                offset_z = np.random.randint(dz - z + 1)
                offset_y = np.random.randint(dy - y + 1)
                offset_x = np.random.randint(dx - x + 1)
                data[key] = value[:, :, offset_z:offset_z + z, offset_y:offset_y + y, offset_x:offset_x + x]
        return data





