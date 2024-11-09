import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out = torch.max(x, dim=(2, 3), keepdim=True)[0]
        out = self.fc1(avg_out) + self.fc1(max_out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out) * x

class FeatureSimilarityLoss(nn.Module):
    def __init__(self, in_channels):
        super(FeatureSimilarityLoss, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, fused_features, features1, features2):
        enhanced_feat1 = self.ca(features1)
        enhanced_feat2 = self.ca(features2)
        return self.weight * F.mse_loss(fused_features, enhanced_feat1 + enhanced_feat2)

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15], pretrained=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=pretrained).features
        self.layers = [vgg[i] for i in layers]
        for layer in self.layers:
            layer.requires_grad = False

    def forward(self, output, target):
        loss = 0
        x, y = output, target
        for layer in self.layers:
            x, y = layer(x), layer(y)
            loss += F.mse_loss(x, y)
        return loss

class TVLoss(nn.Module):
    def forward(self, x):
        return torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + \
               torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))

class CompositeLoss(nn.Module):
    def __init__(self, in_channels, alpha1=1.0, alpha2=1.0, beta=1.0):
        super(CompositeLoss, self).__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.feature_similarity_loss = FeatureSimilarityLoss(in_channels)
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TVLoss()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta

    def forward(self, output, target, features_output, features1, features2):
        L_det = 1 - self.ms_ssim(output, target)
        L_feat = self.feature_similarity_loss(features_output, features1, features2)
        L_perceptual = self.perceptual_loss(output, target)
        L_tv = self.tv_loss(output)
        L_fuse = L_feat + self.alpha1 * L_det + self.alpha2 * L_perceptual + self.beta * L_tv
        return L_fuse
