"""Train various LSTMs"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from .dataloaders import *
from .neural_nets import *


def train_LSTM(train_data,
               valid_data,
               test_data,
               ap_score,
               device,
               max_epochs=125,
               lr=3e-4,
               no_weight_loss=False,
               best_map=False):
    features, weak_labels, Y = train_data
    valid_features, valid_weak_labels, valid_Y = valid_data
    test_features, test_weak_labels, test_Y = test_data
    pin_memory = device != 'cpu'

    ds = LSTMDataset(features, weak_labels, Y, no_weight_loss=no_weight_loss)

    dl = DataLoader(
        ds,
        shuffle=True,
        batch_size=32,
        num_workers=6,
        pin_memory=pin_memory,
    )

    valid_ds = LSTMDataset(valid_features, valid_weak_labels, valid_Y)
    valid_dl = DataLoader(valid_ds,
                          shuffle=False,
                          batch_size=len(valid_ds),
                          num_workers=6,
                          pin_memory=pin_memory)

    model = LSTMClassifierSkip(
        ds.X.shape[-1],
        ds.weak_labels.shape[-1] if weak_labels is not None else 0,
        ds.num_classes,
    ).to(device)

    CELoss = nn.CrossEntropyLoss(weight=ds.weights.to(device))
    optimizer = Adam(model.parameters(), lr=lr)

    best_valid = 0
    best_metric = 0 if best_map else float('inf')
    best_ap = None

    for epoch in range(max_epochs):
        # train
        model.train()
        for idx, data in enumerate(dl):
            optimizer.zero_grad()

            if weak_labels is not None:
                x, wl, y = data
                x = x.to(device, non_blocking=True).float()
                wl = wl.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).long()
                inputs = (x, wl)
            else:
                x, y = data
                x = x.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).long()
                inputs = (x, )

            pred = torch.squeeze(model(*inputs))
            loss = CELoss(pred, y)
            loss.backward()
            optimizer.step()

        # valid
        model.eval()
        if valid_weak_labels is not None:
            x, wl, y = next(iter(valid_dl))
            x = x.to(device, non_blocking=True).float()
            wl = wl.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()
            inputs = (x, wl)
        else:
            x, y = next(iter(valid_dl))
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()
            inputs = (x, )

        pred = torch.squeeze(model(*inputs))
        if best_map:
            ap_score_ = ap_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            print('ap_score', ap_score_)
            if ap_score_ > best_metric:
                best_metric = ap_score_
                best_valid = copy.deepcopy(model.state_dict())
        else:
            valid_loss = CELoss(pred, y)
            print('loss', valid_loss.detach().cpu().numpy())
            if valid_loss < best_metric:
                best_metric = valid_loss
                best_valid = copy.deepcopy(model.state_dict())

    print('BEST', best_metric)

    model.load_state_dict(best_valid)

    # get test AP
    test_ds = LSTMDataset(test_features, test_weak_labels, test_Y)

    test_dl = DataLoader(test_ds,
                         shuffle=False,
                         batch_size=len(test_ds),
                         num_workers=6,
                         pin_memory=pin_memory)
    model.eval()
    if test_weak_labels is not None:
        x, wl, y = next(iter(test_dl))
        x = x.to(device, non_blocking=True).float()
        wl = wl.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        inputs = (x, wl)
    else:
        x, y = next(iter(test_dl))
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        inputs = (x, )

    pred = torch.squeeze(model(*inputs))
    best_ap = ap_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())

    return model.cpu(), best_ap


def train_LSTM_student(
    train_data,
    valid_data,
    test_data,
    ap_score,
    sel_inds,
    return_probs,
    device='cpu',
    max_epochs=50,
    lr=3e-4,
    no_weight_loss=False,
    best_map=False,
):
    features, Y = train_data
    valid_features, valid_Y = valid_data
    test_features, test_Y = test_data
    pin_memory = device != 'cpu'

    # single frame train ds
    ds = LSTMDataset(features[sel_inds], None, Y[sel_inds], no_weight_loss=no_weight_loss)
    dl = DataLoader(ds, shuffle=True, batch_size=32, num_workers=6, pin_memory=pin_memory)

    # single frame valid ds
    valid_ds = LSTMDataset(valid_features, None, valid_Y)
    valid_dl = DataLoader(valid_ds,
                          shuffle=False,
                          batch_size=len(valid_ds),
                          num_workers=6,
                          pin_memory=pin_memory)

    model = LSTMClassifierStudent(ds.X.shape[-1], ds.num_classes).to(device)

    Softmax_dim1 = nn.Softmax(dim=1)
    CELoss = nn.CrossEntropyLoss(weight=ds.weights.to(device))
    optimizer = Adam(model.parameters(), lr=lr)

    best_metric = 0 if best_map else float('inf')
    best_ap = None

    for epoch in range(max_epochs):
        # train
        model.train()
        for idx, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()
            optimizer.zero_grad()
            pred = torch.squeeze(model(x))
            loss = CELoss(pred, y)
            loss.backward()
            optimizer.step()

        # valid
        model.eval()
        x, y = next(iter(valid_dl))
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        pred = torch.squeeze(model(x))
        if best_map:
            ap_score_ = ap_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
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

    model.load_state_dict(best_valid)

    preds = []
    for f in [features, valid_features, test_features]:
        test_ds = LSTMDataset(f, None, None)
        test_dl = DataLoader(test_ds,
                             shuffle=False,
                             batch_size=len(test_ds),
                             num_workers=8,
                             pin_memory=pin_memory)
        model.eval()
        x, = next(iter(test_dl))
        x = x.to(device, non_blocking=True).float()
        pred = Softmax_dim1(torch.squeeze(model(x)).detach().cpu()).numpy()
        if not return_probs:
            pred = np.expand_dims(np.argmax(pred, 1), -1)
        preds.append(pred)

    test_ds = ActiveLearningNNDataset(test_features, None, test_Y)
    test_dl = DataLoader(test_ds,
                         shuffle=False,
                         batch_size=len(test_ds),
                         num_workers=6,
                         pin_memory=pin_memory)
    model.eval()
    x, y = next(iter(test_dl))
    x = x.to(device, non_blocking=True).float()
    y = y.to(device, non_blocking=True).long()
    pred = torch.squeeze(model(x))
    best_ap = ap_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())

    return preds, best_ap


def get_probs(model, features, weak_labels, device):
    Softmax_dim1 = nn.Softmax(dim=1)
    model = model.to(device)
    model.eval()
    ds = LSTMDataset(features, weak_labels, None)
    dl = DataLoader(ds, shuffle=False, batch_size=len(ds))
    if weak_labels is not None:
        total_data, wl = next(iter(dl))
        total_data = total_data.to(device).float()
        wl = wl.to(device).float()
        inputs = (total_data, wl)
    else:
        total_data, = next(iter(dl))
        total_data = total_data.to(device).float()
        inputs = (total_data, )

    preds = Softmax_dim1(torch.squeeze(model(*inputs)))
    preds = preds.detach().cpu().numpy()
    return preds
