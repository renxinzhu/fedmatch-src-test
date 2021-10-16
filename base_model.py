from typing import Dict
from layers import DConv2d, DLinear, DModule
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(DModule):
    def __init__(self):
        super().__init__()

        def conv_with_relu(*args, **kargs):
            return nn.Sequential(
                DConv2d(*args, **kargs),
                nn.ReLU(),
            )

        self.max_pool = nn.MaxPool2d(2, 2)
        self.last_max_pool = nn.MaxPool2d(4, 4)
        self.flatten = nn.Flatten()

        self.cnn_block1 = nn.Sequential(
            conv_with_relu(3, 32, 3, padding=1),
            conv_with_relu(32, 64, 3, padding=1),
        )

        self.cnn_block2 = nn.Sequential(
            conv_with_relu(64, 64, 3, padding=1),
            conv_with_relu(64, 64, 3, padding=1),
        )

        self.cnn_block3 = conv_with_relu(64, 128, 3, padding=1)

        self.cnn_block4 = conv_with_relu(128, 256, 3, padding=1)

        self.cnn_block5 = nn.Sequential(
            conv_with_relu(256, 256, 3, padding=1),
            conv_with_relu(256, 256, 3, padding=1),
        )

        self.last_max_pooling = nn.MaxPool2d(4, 1)
        self.fcl = DLinear(256, 10)

        self.to(device)

    def forward(self, X):
        out = self.cnn_block1(X)
        out = self.max_pool(out)

        out = self.cnn_block2(out) + out
        out = self.cnn_block3(out)
        out = self.max_pool(out)

        out = self.cnn_block4(out)
        out = self.max_pool(out)

        out = self.cnn_block5(out) + out
        out = self.last_max_pool(out)

        out = self.fcl(self.flatten(out))

        return out


class Backbone(Model):
    def validate(self, dataloader):
        correct = 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = self.forward(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        return correct / len(dataloader.dataset)


if __name__ == '__main__':
    model = Model()
    import pdb
    pdb.set_trace()

    nps = model.named_parameters()
