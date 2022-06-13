"""Experimental discriminative modesl for weak supervision exps."""
import numpy as np
from . import train_NN, train_LSTM


# all functions must return samples requested and best AP
def clip_probs(probs, thresh=1e-5):
    return np.clip(probs, thresh, 1 - thresh)


def train_discrim_model(trainX,
                        trainY,
                        trainWL,
                        validX,
                        validY,
                        validWL,
                        testX,
                        testY,
                        testWL,
                        ap_score,
                        sel_inds,
                        device='cpu',
                        no_weight_loss=False,
                        best_map=False,
                        is_seq=False):
    if trainWL is not None and validWL is not None and testWL is not None:
        trainWL = np.concatenate(trainWL, -1)
        validWL = np.concatenate(validWL, -1)
        testWL = np.concatenate(testWL, -1)

    trainY = np.squeeze(trainY)
    validY = np.squeeze(validY)
    testY = np.squeeze(testY)

    train_fn = train_LSTM.train_LSTM if is_seq else train_NN.train_NN
    get_probs_fn = train_LSTM.get_probs if is_seq else train_NN.get_probs

    model, _ = train_fn((trainX[sel_inds], trainWL[sel_inds], trainY[sel_inds]),
                        (validX, validWL, validY), (testX, testWL, testY),
                        ap_score,
                        device,
                        no_weight_loss=no_weight_loss,
                        best_map=best_map,
                        max_epochs=100)

    train_preds = get_probs_fn(model, trainX, trainWL, device)

    return train_preds
