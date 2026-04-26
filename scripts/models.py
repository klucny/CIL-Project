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

    def compute_loss(self, pred, target, eps = 1e-9) -> torch.Tensor:
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

        #Encoder
        self.enc_conv1 = self._double_conv(3, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc_conv2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc_conv3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.bottleneck = self._double_conv(256, 512)

        # Decoder
        self.up_conv3 = nn.ConvTranspose2d(512, 256, stride=2, kernel_size=2)
        self.dec_conv3 = self._double_conv(512, 256)

        self.up_conv2 = nn.ConvTranspose2d(256, 128, stride=2, kernel_size=2)
        self.dec_conv2 = self._double_conv(256, 128)

        self.up_conv1 = nn.ConvTranspose2d(128, 64, stride=2, kernel_size=2)
        self.dec_conv1 = self._double_conv(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)


    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x) -> torch.Tensor:
        e1 = self.enc_conv1(x)
        p1 = self.pool1(e1)

        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up_conv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec_conv3(d3)

        d2 = self.up_conv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.up_conv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_conv1(d1)

        out = self.out_conv(d1)
        out = F.softplus(out) + 1e-4

        return out.squeeze(1)

    # def eval(self,x):
    #     with torch.no_grad():
    #         return self.forward(x)


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

    # def eval(self, x):
    #     with torch.no_grad():
    #         return self.forward(x)