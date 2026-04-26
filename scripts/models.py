import torch
from torch import nn
import torch.nn.functional as F


# Class Net just acts as a superclass for clean typing
# When creating new Models, inherit from Net and implement the methods
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self,x) -> torch.Tensor:
        return x

    def compute_loss(self, pred, target, eps = 1e-7) -> torch.Tensor:


        gt_mask = (target > eps)

        num_valid_pixels = torch.sum(gt_mask)

        if num_valid_pixels == 0:
            raise Exception("No valid pixels in given image, cannot compute loss")

        preds_safe = torch.clamp(pred, min=eps)

        log_gt_filtered = torch.log(target[gt_mask])
        log_pred_filtered = torch.log(preds_safe[gt_mask])

        diffs : torch.Tensor = log_pred_filtered - log_gt_filtered

        alpha: torch.Tensor = torch.mean(-diffs)

        loss : torch.Tensor =  torch.sqrt(torch.mean(torch.pow(alpha + diffs,2)))

        return loss


class CNN(Net):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=2)
        self.fc1 = nn.Linear(64 * 66 * 66 , 2*2)

        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.norm_d1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.norm_d2 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.norm_d3 = nn.BatchNorm2d(8)
        self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=33)

        self.decoder = nn.Sequential(
            self.deconv1,
            self.norm_d1,
            self.relu,
            self.deconv2,
            self.norm_d2,
            self.relu,
            self.deconv3,
            self.norm_d3,
            self.relu,
            self.deconv4
        )

        #used to only get positive values, otherwise issues might arise when computing the logarithm in the loss function
        self.soft_plus = nn.Softplus()



    def forward(self, x) -> torch.Tensor:

        # Encoder
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.maxpool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.maxpool1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.decoder(x)

        x = self.soft_plus(x)

        return x.squeeze(1)

    def eval(self,x):
        with torch.no_grad():
            return self.forward(x)


class CNNSmall(Net):
    def __init__(self) -> None:
        super(CNNSmall, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 4)
        self.norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2)

        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=7, stride=3, padding=7)
        self.norm_d1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=7, stride=3, padding=7)
        self.norm_d2 = nn.BatchNorm2d(8)
        self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=7)

        self.decoder = nn.Sequential(
            self.deconv1,
            self.norm_d1,
            self.relu,
            self.deconv2,
            self.norm_d2,
            self.relu,
            self.deconv4
        )

    def forward(self, x) -> torch.Tensor:
        # Encoder
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.maxpool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.maxpool1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.decoder(x)

        return x.squeeze(1)

    def eval(self, x):
        with torch.no_grad():
            return self.forward(x)