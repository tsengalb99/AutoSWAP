import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from .dataloaders import *
from .neural_nets import *
from .loss import SoftCrossEntropyLoss


def train_NN_self_supervision(train_data,
                              valid_data,
                              test_data,
                              ap_score,
                              device,
                              max_epochs=125,
                              lr=3e-4,
                              no_weight_loss=False,
                              best_map=False,
                              override_weights=None):
    features, Y = train_data
    valid_features, valid_Y = valid_data
    test_features, test_Y = test_data

    pin_memory = device != 'cpu'

    train_ds = ActiveLearningNNDataset(features, None, Y, no_weight_loss=no_weight_loss)

    if override_weights is not None:
        train_ds.weights = override_weights

    valid_ds = ActiveLearningNNDataset(valid_features, None, valid_Y)
    test_ds = ActiveLearningNNDataset(test_features, None, test_Y)

    train_dl = DataLoader(train_ds,
                          shuffle=True,
                          batch_size=32,
                          num_workers=6,
                          pin_memory=pin_memory)
    valid_dl = DataLoader(valid_ds,
                          shuffle=True,
                          batch_size=len(valid_ds),
                          num_workers=12,
                          pin_memory=pin_memory)
    test_dl = DataLoader(test_ds,
                         shuffle=True,
                         batch_size=len(test_ds),
                         num_workers=12,
                         pin_memory=pin_memory)

    model = FrameNN(train_ds.X.shape[-1], train_ds.num_classes, 256, 0.4).to(device)

    soft_labels = len(Y.shape) > 1 and Y.shape[-1] != 1

    if soft_labels:
        CELoss = SoftCrossEntropyLoss(weight=train_ds.weights.to(device), reduction='mean')
    else:
        CELoss = nn.CrossEntropyLoss(weight=train_ds.weights.to(device), reduction='mean')

    optimizer = Adam(model.parameters(), lr=lr)

    best_model_state = 0
    best_metric = 0 if best_map else float('inf')
    best_ap = None

    for epoch in range(max_epochs):
        print(f'EPOCH {epoch}')
        # train
        model.train()
        for idx, data in enumerate(train_dl):
            optimizer.zero_grad()

            #x, y, batch_weights = data
            x, y = data
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            y = y.float() if soft_labels else y.long()
            #batch_weights = batch_weights.to(device, non_blocking=True).float()
            inputs = (x, )

            pred = torch.squeeze(model(*inputs))
            loss = CELoss(pred, y)
            #loss = torch.sum(loss * batch_weights) / len(batch_weights)
            loss.backward()
            optimizer.step()

        model.eval()

        # valid preds
        x, y = next(iter(valid_dl))
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        y = y.float() if soft_labels else y.long()
        inputs = (x, )
        pred = torch.squeeze(model(*inputs))

        if best_map:
            y_hard = np.argmax(y.detach().cpu().numpy(), 1)
            ap_score_ = ap_score(y_hard, pred.detach().cpu().numpy())

            print('valid ap score', ap_score_)
            if ap_score_ > best_metric:
                best_metric = ap_score_
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            valid_loss = torch.mean(CELoss(pred, y))
            print('valid loss', valid_loss.detach().cpu().numpy())
            if valid_loss < best_metric:
                best_metric = valid_loss
                best_model_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            # test preds
            x, y = next(iter(test_dl))
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            y = y.float() if soft_labels else y.long()
            inputs = (x, )
            pred = torch.squeeze(model(*inputs))
            if best_map:
                y_hard = np.argmax(y.detach().cpu().numpy(), 1)
                ap_score_ = ap_score(y_hard, pred.detach().cpu().numpy())
                print('test ap score', ap_score_)
            else:
                test_loss = torch.mean(CELoss(pred, y))
                print('test loss', test_loss.detach().cpu().numpy())

    print('BEST', best_metric)

    model.load_state_dict(best_model_state)
    model.eval()
    x, y = next(iter(test_dl))
    x = x.to(device, non_blocking=True).float()
    y = y.to(device, non_blocking=True)
    y = y.float() if soft_labels else y.long()
    inputs = (x, )

    y_hard = np.argmax(y.detach().cpu().numpy(), 1)
    pred = torch.squeeze(model(*inputs))
    best_ap = ap_score(y_hard, pred.detach().cpu().numpy())

    return model.cpu(), best_ap
