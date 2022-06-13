"""Various datasets"""
import numpy as np
import torch
from torch.utils.data import Dataset


class ActiveLearningNNDataset(Dataset):
    def __init__(self, features, weak_labels, Y, sample_weights=None, no_weight_loss=False):
        self.features = features
        self.weak_labels = weak_labels

        self.Y = Y
        if self.Y is not None:
            soft_labels = len(Y.shape) > 1 and Y.shape[-1] != 1

            if soft_labels:
                self.num_classes = Y.shape[-1]
            else:
                self.num_classes = int(np.max(self.Y.flatten())) + 1

            if not no_weight_loss:
                if soft_labels:
                    counts = np.sum(self.Y, 0)
                else:
                    _, counts = np.unique(self.Y, return_counts=True)

                print('CLASS DISTRIBUTION', counts / sum(counts))

                self.weights = np.sqrt(1 / counts)
                self.weights = torch.tensor(self.weights / sum(self.weights)).float()
            else:
                self.weights = torch.tensor(np.ones(self.num_classes) / self.num_classes).float()
            print(self.weights)

        self.sample_weights = sample_weights
        self.X = self.features

        if self.weak_labels is not None:
            self.weak_labels = np.expand_dims(self.weak_labels, 1)
            self.X = np.concatenate([self.X, self.weak_labels], 2)

        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ret_arr = [self.X[idx]]
        if self.Y is not None:
            ret_arr.append(self.Y[idx])
        if self.sample_weights is not None:
            ret_arr.append(self.sample_weights[idx])
        return tuple(ret_arr) if len(ret_arr) > 1 else ret_arr[0]


class ActiveLearningNNDatasetSkip(Dataset):
    def __init__(self, features, weak_labels, Y, sample_weights=None, no_weight_loss=False):
        self.features = features
        self.weak_labels = weak_labels

        self.Y = Y
        if self.Y is not None:
            self.num_classes = int(np.max(self.Y.flatten())) + 1

            if not no_weight_loss:
                _, counts = np.unique(self.Y, return_counts=True)
                self.weights = np.sqrt(1 / counts)
                self.weights = torch.tensor(self.weights / sum(self.weights)).float()
            else:
                self.weights = torch.tensor(np.ones(self.num_classes) / self.num_classes).float()
            print(self.weights)

        self.sample_weights = sample_weights
        self.X = self.features

        if self.weak_labels is not None:
            self.weak_labels = np.expand_dims(self.weak_labels, 1)

        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ret_arr = [self.X[idx], self.weak_labels[idx]]
        if self.Y is not None:
            ret_arr.append(self.Y[idx])
        if self.sample_weights is not None:
            ret_arr.append(self.sample_weights[idx])
        return tuple(ret_arr)


class LSTMDataset(Dataset):
    def __init__(self, X, weak_labels, Y, sample_weights=None, no_weight_loss=False):
        self.X = X
        self.weak_labels = weak_labels

        self.Y = Y
        if self.Y is not None:
            self.num_classes = int(np.max(self.Y.flatten())) + 1

            if not no_weight_loss:
                _, counts = np.unique(self.Y, return_counts=True)
                self.weights = np.sqrt(1 / counts)
                self.weights = torch.tensor(self.weights / sum(self.weights)).float()
            else:
                self.weights = torch.tensor(np.ones(self.num_classes) / self.num_classes).float()
            print(self.weights)

        self.sample_weights = sample_weights

        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ret_arr = [self.X[idx]]
        if self.weak_labels is not None:
            ret_arr.append(self.weak_labels[idx])
        if self.Y is not None:
            ret_arr.append(self.Y[idx])
        if self.sample_weights is not None:
            ret_arr.append(self.sample_weights[idx])
        return tuple(ret_arr)


class LSTMDatasetSoftLabels(Dataset):
    def __init__(self, X, weak_labels, Y, sample_weights=None, no_weight_loss=False):
        self.X = X
        self.weak_labels = weak_labels

        self.Y = Y
        if self.Y is not None:
            self.num_classes = Y.shape[-1]
            if not no_weight_loss:
                counts = np.sum(Y, 0)
                self.weights = np.sqrt(1 / counts)
                self.weights = torch.tensor(self.weights / sum(self.weights)).float()
            else:
                self.weights = torch.tensor(np.ones(self.num_classes) / self.num_classes).float()
            print(self.weights)

        self.sample_weights = sample_weights

        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ret_arr = [self.X[idx]]
        if self.weak_labels is not None:
            ret_arr.append(self.weak_labels[idx])
        if self.Y is not None:
            ret_arr.append(self.Y[idx])
        if self.sample_weights is not None:
            ret_arr.append(self.sample_weights[idx])
        return tuple(ret_arr)
