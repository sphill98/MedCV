import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        output = output.contiguous()
        target = target.contiguous()

        intersection = (output * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + self.smooth) / (output.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))

        return loss.mean()
    
