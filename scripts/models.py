import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as torch_models
from torchvision.models.resnet import conv1x1
import copy


# Class Net just acts as a superclass for clean typing
# When creating new Models, inherit from Net and implement the methods
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # sobel filters for gradient computation
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[[ -1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x) -> torch.Tensor:
        return x

    def compute_loss(self, pred, target, grad_weight=0.0, eps=1e-9) -> tuple:
        gt_mask = (target > eps)

        num_valid_pixels = torch.sum(gt_mask)

        if num_valid_pixels == 0:
            raise Exception("No valid pixels in given image, cannot compute loss")

        preds_safe = torch.clamp(pred, min=eps)
        target_safe = torch.clamp(target, min=eps)

        log_gt_filtered = torch.log(target[gt_mask])
        log_pred_filtered = torch.log(preds_safe[gt_mask])

        diffs: torch.Tensor = log_pred_filtered - log_gt_filtered

        alpha: torch.Tensor = torch.mean(-diffs)

        sirmse_loss: torch.Tensor = torch.sqrt(torch.mean(torch.pow(alpha + diffs, 2)))

        if grad_weight <= 0.0:
            return sirmse_loss, sirmse_loss, torch.tensor(0.0)

        # Gradients
        pred_padded = F.pad(torch.log(preds_safe).unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        target_padded = F.pad(torch.log(target_safe).unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        gradient_x_predicted = F.conv2d(pred_padded, self.sobel_x)
        gradient_y_predicted = F.conv2d(pred_padded, self.sobel_y)
        gradient_x_groundtruth = F.conv2d(target_padded, self.sobel_x)
        gradient_y_groundtruth = F.conv2d(target_padded, self.sobel_y)

        # compute gradient loss
        gradient_diff_x = torch.abs(gradient_x_predicted - gradient_x_groundtruth)
        gradient_diff_y = torch.abs(gradient_y_predicted - gradient_y_groundtruth)

        gradient_loss = torch.mean(gradient_diff_x[gt_mask.unsqueeze(1)]) + torch.mean(gradient_diff_y[gt_mask.unsqueeze(1)])

        total_loss = sirmse_loss + grad_weight * gradient_loss

        return total_loss, sirmse_loss, gradient_loss


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
        self.up_conv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        )
        self.dec_conv4 = self._double_conv(2048, 1024)

        self.up_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        )
        self.dec_conv3 = self._double_conv(1024, 512)

        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1)
        )
        self.dec_conv2 = self._double_conv(512, 256)

        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1)
        )
        self.dec_conv1 = self._double_conv(128, 64)

        self.up_conv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
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

class Canny(Net):
    pass

class CannyCNN(CNN, Canny):
    def __init__(self) -> None:
        super(CannyCNN, self).__init__()
        # This is a small encoder to process the Canny edge map.
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.fusion_conv = nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1)

    # overload of CNN.forward()
    def forward(self, x, edges) -> torch.Tensor:
        pad_size = 8  # needed to bring the pictures to size 572 because otherwise with ResNet size issue will occur

        x = F.pad(x, [pad_size, pad_size, pad_size, pad_size], mode='reflect')
        edges = F.pad(edges, [pad_size, pad_size, pad_size, pad_size], mode='reflect')

        e1 = self.encoder_conv1(x)
        p1 = self.pool(e1)

        e2 = self.encoder_layer1(p1)
        e3 = self.encoder_layer2(e2)
        e4 = self.encoder_layer3(e3)
        b = self.encoder_layer4(e4)

        edge_features = self.edge_encoder(edges)


        # b = b * edge_features
        b = torch.cat([b, edge_features], dim=1)
        b = self.fusion_conv(b)


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


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # We reduce the channels inside the parallel branches to keep memory usage safe
        mid_channels = 512

        # Branch 1: 1x1 Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 2: 3x3 Convolution, Dilation 6
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 3: 3x3 Convolution, Dilation 12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 4: 3x3 Convolution, Dilation 18
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 5: Global Average Pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Final 1x1 convolution to fuse all branches and restore channel count
        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # The pooled branch needs to be resized back to the spatial dimensions of the other branches
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate all branches along the channel dimension
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.final_conv(out)
        return out


class CannyCNNSkip(CNN, Canny):
    def __init__(self) -> None:
        super(CannyCNNSkip, self).__init__()

        # edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(2048, 2048)
        self.attention_gate = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1), # e1 has 64 channels, edge_features has 64
            nn.Sigmoid()
        )

    def forward(self, c, edges) -> torch.Tensor:
        pad_size = 8

        c = F.pad(c, [pad_size, pad_size, pad_size, pad_size], mode='reflect')
        edges = F.pad(edges, [pad_size, pad_size, pad_size, pad_size], mode='reflect')

        # standard RGB encoder
        e1 = self.encoder_conv1(c)
        p1 = self.pool(e1)

        e2 = self.encoder_layer1(p1)
        e3 = self.encoder_layer2(e2)
        e4 = self.encoder_layer3(e3)
        b = self.encoder_layer4(e4)

        # ASPP
        b = self.aspp(b)

        # edge features skip connection
        edge_features = self.edge_encoder(edges)
        concat_features = torch.cat([e1, edge_features], dim=1)
        attn_mask = self.attention_gate(concat_features)
        e1 = e1 + (edge_features * attn_mask) # Only add the edges the network thinks are useful!

        # standard decoder
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



class ViT(Net):
    def __init__(self) -> None:
        super(ViT, self).__init__()
        pass

