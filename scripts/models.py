import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as torch_models


# Class Net just acts as a superclass for clean typing
# When creating new Models, inherit from Net and implement the methods
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x) -> torch.Tensor:
        return x

    def compute_loss(self, pred, target, eps=1e-9) -> torch.Tensor:
        gt_mask = (target > eps)

        num_valid_pixels = torch.sum(gt_mask)

        if num_valid_pixels == 0:
            raise Exception("No valid pixels in given image, cannot compute loss")

        preds_safe = torch.clamp(pred, min=eps)

        log_gt_filtered = torch.log(target[gt_mask])
        log_pred_filtered = torch.log(preds_safe[gt_mask])

        diffs: torch.Tensor = log_pred_filtered - log_gt_filtered

        alpha: torch.Tensor = torch.mean(-diffs)

        loss: torch.Tensor = torch.sqrt(torch.mean(torch.pow(alpha + diffs, 2)))

        return loss


class CNN(Net):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        # Use the recommended 'weights' parameter instead of deprecated 'pretrained=True'
        resnet = torch_models.resnet50(weights=torch_models.ResNet50_Weights.DEFAULT)

        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        self.pool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        # Decoder
        self.up_conv4 = nn.ConvTranspose2d(2048, 1024, stride=2, kernel_size=2)
        self.dec_conv4 = self._double_conv(2048, 1024)

        self.up_conv3 = nn.ConvTranspose2d(1024, 512, stride=2, kernel_size=2)
        self.dec_conv3 = self._double_conv(1024, 512)

        self.up_conv2 = nn.ConvTranspose2d(512, 256, stride=2, kernel_size=2)
        self.dec_conv2 = self._double_conv(512, 256)

        self.up_conv1 = nn.ConvTranspose2d(256, 64, stride=2, kernel_size=2)
        self.dec_conv1 = self._double_conv(128, 64)

        self.up_conv0 = nn.ConvTranspose2d(64, 32, stride=2, kernel_size=2)
        self.dec_conv0 = self._double_conv(32, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

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
        pad_size = 8 # needed to bring the pictures to size 572 because otherwise with ResNet size issue will occur

        x = F.pad(x, [pad_size, pad_size, pad_size, pad_size], mode='reflect')

        e1 = self.encoder_conv1(x)
        p1 = self.pool(e1)

        e2 = self.encoder_layer1(p1)
        e3 = self.encoder_layer2(e2)
        e4 = self.encoder_layer3(e3)
        b = self.encoder_layer4(e4)

        d4 = self.up_conv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec_conv4(d4)

        d3 = self.up_conv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec_conv3(d3)

        d2 = self.up_conv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.up_conv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_conv1(d1)

        d0 = self.up_conv0(d1)
        d0 = self.dec_conv0(d0)

        out = self.out_conv(d0)
        out = F.softplus(out) + 1e-4

        return out.squeeze(1)[:, pad_size:-pad_size, pad_size:-pad_size]


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

    class Transformer(Net):
        pass
