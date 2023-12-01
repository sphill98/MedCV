import torch

def dice_loss(output: torch.Tensor, target: torch.Tensor, smooth=1.):
    output = output.contiguous()
    target = target.contiguous()

    intersection = (output * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (output.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()