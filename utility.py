import os
import torch
import shutil
import numpy as np


def create_dir(dir, opts=None):
    try:
        if os.path.exists(dir):
            if opts == 'del':
                shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)
    except OSError:
        print("Error: Failed to create the directory.")


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


def load(ckpt_dir, net, optim, set_epoch=None):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # numbers = re.findall(r'\d+', string)
    if set_epoch is None:
         dict_model = torch.load('%s%s' % (ckpt_dir, ckpt_lst[-1]))
    else:
        dict_model = torch.load('%s%s' % (ckpt_dir, 'model_epoch'+str(set_epoch)+'.pth'))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    if set_epoch is None:
        epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    else:
        epoch = set_epoch

    return net, optim, epoch


class Normaliz():
    def __init__(self, x, window=None):
        self.meanx = np.mean(x) if window is None else window[0]
        self.stdx = np.std(x) if window is None else window[1]

    def Normal(self, x):
        return (x - self.meanx) / self.stdx

    def Denorm(self, x):
        return (x * self.stdx) + self.meanx
   