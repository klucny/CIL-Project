import torch
from torch import nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4)


    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def eval(self,x):
        with torch.no_grad():
            ...

    def compute_loss(self,x,y):
        loss = F.mse_loss(x,y)
        return loss

