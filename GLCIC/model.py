import torch
import torch.nn as nn
import torch.nn.functional as F


class CompletionNetwork(nn.Module):
    def __init__(self):
        super(CompletionNetwork, self).__init__()

        def down_conv(num_in_feature, num_out_feature, kernel_size=5, stride=1, padding=2, dilation=0, normalize=True):
            layers = [nn.Conv2d(num_in_feature, num_out_feature, kernel_size=kernel_size, dilation=dilation,
                                stride=stride, padding=padding), nn.ReLU()]
            if normalize:
                layers.append(nn.BatchNorm2d(num_out_feature))
            return layers

        def up_conv(num_in_feature, num_out_feature, kernel_size=4, stride=2, padding=1, normalize=True):
            layers = [nn.ConvTranspose2d(num_in_feature, num_out_feature, kernel_size=kernel_size,
                                         stride=stride, padding=padding), nn.ReLU()]
            if normalize:
                layers.append(nn.BatchNorm2d(num_out_feature))
            return layers

        self.layers = nn.Sequential(
            *down_conv(4, 64, 5, 1, 2),
            *down_conv(64, 128, 3, 2, 1),
            *down_conv(128, 128, 3, 1, 1),
            *down_conv(128, 256, 3, 2, 1),
            *down_conv(256, 256, 3, 1, 1),
            *down_conv(256, 256, 3, 1, 1),
            *down_conv(256, 256, 3, 1, 2, 2),
            *down_conv(256, 256, 3, 1, 4, 4),
            *down_conv(256, 256, 3, 1, 8, 8),
            *down_conv(256, 256, 3, 1, 16, 16),
            *down_conv(256, 256, 3, 1, 1),
            *down_conv(256, 256, 3, 1, 1),
            *up_conv(256, 128),
            *down_conv(128, 128, 3, 1, 1),
            *up_conv(128, 64),
            *down_conv(64, 32, 3, 1, 1),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

        def forward(self, x):
            return self.layers(x)


class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
