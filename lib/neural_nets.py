import random
import torch
import torch.nn as nn
from .dataloaders import *


class FrameNN(nn.Module):
    def __init__(self, in_dim, out_dim, max_dim=128, max_dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, max_dim),
            nn.ReLU(),
            nn.Dropout(max_dropout),
            nn.Linear(max_dim, max_dim // 2),
            nn.ReLU(),
            nn.Dropout(max_dropout - 0.1),
            nn.Linear(max_dim // 2, max_dim // 4),
            nn.ReLU(),
            nn.Dropout(max_dropout - 0.2),
            nn.Linear(max_dim // 4, out_dim),
        )

    def forward(self, features):
        return self.fc(features)


class FrameNNSkip(nn.Module):
    def __init__(self, feature_dim, wl_dim, out_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(feature_dim + wl_dim, 128),
            nn.Linear(128 + wl_dim, 64),
            nn.Linear(64 + wl_dim, 32),
            nn.Linear(32 + wl_dim, out_dim)
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5),
            nn.Dropout(0.4),
            nn.Dropout(0.3),
        ])

        self.activation = nn.ReLU()

    def forward(self, features, weak_labels):
        x = features
        for i in range(len(self.layers)):
            x = torch.cat([x, weak_labels], dim=2)
            x = self.layers[i](x)
            if i == len(self.layers) - 1:
                break
            x = self.activation(x)
            x = self.dropouts[i](x)

        return x


class FrameNNStudent(nn.Module):
    # weak student model
    def __init__(self, in_dim, out_dim):
        super().__init__()
        L0 = int(random.random() * 50) + 40
        L1 = int(random.random() * 50) + 30
        L2 = int(random.random() * 20) + 10
        D0 = random.random() * 0.20
        D1 = random.random() * 0.20
        D2 = random.random() * 0.10
        print('Layer sizes \t {}'.format([L0, L1, L2]))
        print('Dropout vals\t {}'.format([D0, D1, D2]))
        self.fc = nn.Sequential(
            nn.Linear(in_dim, L0),
            nn.ReLU(),
            nn.Dropout(D0),
            nn.Linear(L0, L1),
            nn.ReLU(),
            nn.Dropout(D1),
            nn.Linear(L1, L2),
            nn.ReLU(),
            nn.Dropout(D2),
            nn.Linear(L2, out_dim),
        )

    def forward(self, features):
        return self.fc(features)


class LSTMClassifierStudent(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hs = int(random.random() * 64) + 64
        nl = int(random.random() * 2) + 1
        self.LSTM = nn.LSTM(
            input_size=in_dim,
            hidden_size=hs,
            num_layers=nl,
            dropout=random.random() * 0.20 if nl > 1 else 0,
            batch_first=True,
        )

        L0 = int(random.random() * 50) + 30
        L1 = int(random.random() * 20) + 10
        D0 = random.random() * 0.20
        D1 = random.random() * 0.20

        self.fc = nn.Sequential(
            nn.Linear(hs * nl, L0),
            nn.ReLU(),
            nn.Dropout(D0),
            nn.Linear(L0, L1),
            nn.ReLU(),
            nn.Dropout(D1),
            nn.Linear(L1, out_dim),
        )

    def forward(self, features):
        _, (LSTM_out, _) = self.LSTM(features)
        # convert from (D*nL, T, H) to (T, D*nL, h)
        LSTM_out = torch.transpose(LSTM_out, 0, 1)
        LSTM_out = LSTM_out.flatten(start_dim=1)
        return self.fc(LSTM_out)


class LSTMClassifierSkip(nn.Module):
    def __init__(self, in_dim, wl_dim, out_dim):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=in_dim,
                            hidden_size=128,
                            num_layers=2,
                            dropout=0.2,
                            batch_first=True)

        self.layers = nn.ModuleList([
            nn.Linear(128 * 2 + wl_dim, 128),
            nn.Linear(128 + wl_dim, 48),
            nn.Linear(48 + wl_dim, out_dim),
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.3),
            nn.Dropout(0.2),
        ])

        self.activation = nn.ReLU()

    def forward(self, features, weak_labels=None):
        _, (x, _) = self.LSTM(features)
        # convert from (D*nL, T, H) to (T, D*nL, h)
        x = torch.transpose(x, 0, 1)
        x = x.flatten(start_dim=1)
        for i in range(len(self.layers)):
            if weak_labels is not None:
                x = torch.cat([x, weak_labels], dim=1)
            x = self.layers[i](x)
            if i == len(self.layers) - 1:
                break
            x = self.activation(x)
            x = self.dropouts[i](x)

        return x
