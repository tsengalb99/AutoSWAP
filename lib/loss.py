import torch
import torch.nn as nn
import torch.nn.functional as F


# taken from https://stackoverflow.com/questions/68907809/soft-cross-entropy-in-pytorch
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weight, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.total_wt = torch.sum(self.weight)
        self.reduction_fn = {
            'mean': torch.mean,
            'none': lambda x: x,
            'sum': torch.sum,
        }[reduction]

    def forward(self, y_hat, y):
        p = F.log_softmax(y_hat)
        w_labels = self.weight * y
        loss = torch.sum(-(w_labels * p), 1) / (w_labels.sum())

        return self.reduction_fn(loss)
