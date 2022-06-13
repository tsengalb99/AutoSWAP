import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from .dataloaders import *
from .neural_nets import *
from .loss import SoftCrossEntropyLoss


def train_LSTM_self_supervision(
    train_data,
    valid_data,
    test_data,
    ap_score,
    device='cpu',
    max_epochs=100,
    lr=3e-4,
    no_weight_loss=False,
    best_map=False,
):
    features, Y = train_data
    valid_features, valid_Y = valid_data
    test_features, test_Y = test_data
    pin_memory = device != 'cpu'

    # single frame train ds
    ds = LSTMDatasetSoftLabels(features, None, Y, no_weight_loss=no_weight_loss)
    dl = DataLoader(ds, shuffle=True, batch_size=32, num_workers=6, pin_memory=pin_memory)

    # single frame valid ds
    valid_ds = LSTMDatasetSoftLabels(valid_features, None, valid_Y)
    valid_dl = DataLoader(valid_ds,
                          shuffle=False,
                          batch_size=len(valid_ds),
                          num_workers=6,
                          pin_memory=pin_memory)

    model = LSTMClassifierSkip(ds.X.shape[-1], 0, ds.num_classes).to(device)

    soft_labels = len(Y.shape) > 1 and Y.shape[-1] != 1

    if soft_labels:
        CELoss = SoftCrossEntropyLoss(weight=ds.weights.to(device), reduction='mean')
    else:
        CELoss = nn.CrossEntropyLoss(weight=ds.weights.to(device), reduction='mean')

    optimizer = Adam(model.parameters(), lr=lr)

    best_metric = 0 if best_map else float('inf')
    best_ap = None

    for epoch in range(max_epochs):
        # train
        model.train()
        for idx, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            y = y.float() if soft_labels else y.long()

            optimizer.zero_grad()
            pred = torch.squeeze(model(x))
            loss = CELoss(pred, y)
            loss.backward()
            optimizer.step()

        # valid
        model.eval()
        x, y = next(iter(valid_dl))
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        y = y.float() if soft_labels else y.long()
        pred = torch.squeeze(model(x))
        if best_map:
            y_hard = np.argmax(y.detach().cpu().numpy(), 1)
            ap_score_ = ap_score(y_hard, pred.detach().cpu().numpy())

            print('ap_score', ap_score_)
            if ap_score_ > best_metric:
                best_metric = ap_score_
                best_valid = copy.deepcopy(model.state_dict())
        else:
            valid_loss = CELoss(pred, y)
            print('valid_loss', valid_loss.detach().cpu().numpy())
            if valid_loss < best_metric:
                best_metric = valid_loss
                best_valid = copy.deepcopy(model.state_dict())

    print('BEST', best_metric)

    test_ds = LSTMDatasetSoftLabels(test_features, None, test_Y)
    test_dl = DataLoader(test_ds,
                         shuffle=False,
                         batch_size=len(test_ds),
                         num_workers=6,
                         pin_memory=pin_memory)

    model.load_state_dict(best_valid)
    model.eval()
    x, y = next(iter(test_dl))
    x = x.to(device, non_blocking=True).float()
    y = y.to(device, non_blocking=True)
    y = y.float() if soft_labels else y.long()

    y_hard = np.argmax(y.detach().cpu().numpy(), 1)
    pred = torch.squeeze(model(x))
    best_ap = ap_score(y_hard, pred.detach().cpu().numpy())

    return model.cpu(), best_ap
