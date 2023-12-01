from utility import *
from loadMAT import loadMAT


# Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, plane='sagittal', filename=None, varname=['segTemp', 'rawTemp'], arraySize=256):
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
                    # Write your code here  ex) temp = temp.transpose((0, 1, 2))[None, :, None, :, :]
                    temp = temp.transpose((0, 1, 2))[None, :, None, :, :]
                elif self.plane == 'sagittal':
                    # Write your code here
                    temp = temp.transpose((1, 2, 0))[None, :, None, :, :]
                
                elif self.plane == 'coronal':
                    # Write your code here
                    temp = temp.transpose((2, 1, 0))[None, :, None, :, :]
                else:
                    print('ERROR: plane is not defined')
                    exit()
                datatemp = temp if j==0 else np.concatenate((datatemp, temp), axis=0)
            if i==0 and j==0:  # You can change how the normalize factor is determined. 
                self.minmax = [np.min(datatemp), np.max(datatemp)] 
            if i > 0:
                print(data_array.shape, datatemp.shape)

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
        # Write your code here
        z, y, x = data.shape
        pad_z = max(0, self.arraySize - z)
        pad_y = max(0, self.arraySize - y)
        pad_x = max(0, self.arraySize - x)

        padded_data = np.pad(data, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
        return padded_data


class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data


class RandomRot90(object):
    def __call__(self, data):
        # Write your code here
        k = np.random.randint(4)
        data['label'] = np.rot90(data['label'], k, axes=(1, 2))
        data['input'] = np.rot90(data['input'], k, axes=(1, 2))
        return data


class RandomFlip(object):
    def __call__(self, data):
        # Write your code here
        if np.random.rand() > 0.5:
            data['label'] = np.fliplr(data['label'])
            data['input'] = np.fliplr(data['input'])
        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        # Write your code here
        z, y, x = data['input'].shape
        start_z = np.random.randint(0, z - self.shape[0] + 1)
        start_y = np.random.randint(0, y - self.shape[1] + 1)
        start_x = np.random.randint(0, x - self.shape[2] + 1)

        data['label'] = data['label'][start_z:start_z+self.shape[0], start_y:start_y+self.shape[1], start_x:start_x+self.shape[2]]
        data['input'] = data['input'][start_z:start_z+self.shape[0], start_y:start_y+self.shape[1], start_x:start_x+self.shape[2]]
        return data





