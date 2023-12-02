import torch
import torch.nn as nn

# def dice_loss(output: torch.Tensor, target: torch.Tensor, smooth=1.):
#     output = output.contiguous()
#     target = target.contiguous()

#     intersection = (output * target).sum(dim=2).sum(dim=2)

#     loss = (1 - ((2. * intersection + smooth) / (output.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

#     return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        output = output.contiguous()
        target = output.contiguous()

        intersection = (output * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + self.smooth) / (output.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))

        return loss.mean()
    
