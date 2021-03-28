import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The util functions and layers used in the networks are on the very top of this file.
The body for generator and discriminators are on the later part of this file.
"""


def down_conv(num_in_feature, num_out_feature, kernel_size=5, stride=1, padding=2, dilation=0, normalize=True):
    """
    The convolution module together with batchnorm and activation function.
    """
    layers = [nn.Conv2d(num_in_feature, num_out_feature, kernel_size=kernel_size, dilation=dilation,
                        stride=stride, padding=padding), nn.ReLU()]
    if normalize:
        layers.append(nn.BatchNorm2d(num_out_feature))
    return layers


def up_conv(num_in_feature, num_out_feature, kernel_size=4, stride=2, padding=1, normalize=True):
    """
    The transposed convolution module together with batchnorm and activation function.
    """
    layers = [nn.ConvTranspose2d(num_in_feature, num_out_feature, kernel_size=kernel_size,
                                 stride=stride, padding=padding), nn.ReLU()]
    if normalize:
        layers.append(nn.BatchNorm2d(num_out_feature))
    return layers


class Flatten(nn.Module):
    """
    A layer that flattens an input matrix.
    Used later in discriminators.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Concatenate(nn.Module):
    """
    A layer that concatenates inputs along a certain dimension.
    Used later in discriminators.
    """

    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


"""
Model body:
"""


class CompletionNetwork(nn.Module):
    """
    The generator network.
    """

    def __init__(self):
        super(CompletionNetwork, self).__init__()

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
    """
    Local discriminator network.
    """

    def __init__(self, in_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = in_shape
        self.output_shape = (1024,)
        self.input_channels = in_shape[0]
        self.input_height = in_shape[1]
        self.input_width = in_shape[2]

        self.layers = nn.Sequential(
            *down_conv(self.input_channels, 64, 5, 2, 1),
            *down_conv(64, 128, 5, 2, 2),
            *down_conv(128, 256, 5, 2, 2),
            *down_conv(256, 512, 5, 2, 2),
            *down_conv(512, 512, 5, 2, 2),
            Flatten(),
            nn.Linear(512 * (self.input_height // 32) * (self.input_width // 32), 1024),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class GlobalDiscriminator(nn.Module):
    """
    Global discriminator network.
    """
    def __init__(self, in_shape, architecture='celeba'):
        super(GlobalDiscriminator, self).__init__()
        self.arc = architecture
        self.input_shape = in_shape
        self.out_shape = (1024,)
        self.in_channels = in_shape[0]
        self.in_height = in_shape[1]
        self.in_width = in_shape[2]

        self.layers = nn.Sequential(
            *down_conv(self.in_channels, 64, 5, 2, 2),
            *down_conv(64, 128, 5, 2, 2),
            *down_conv(128, 256, 5, 2, 2),
            *down_conv(256, 512, 5, 2, 2),
            *down_conv(512, 512, 5, 2, 2),
        )

        if self.arc == 'places2':
            self.extra_conv_layer = nn.Sequential(
                *down_conv(512, 512, 5, 2, 2)
            )
            in_channels = 512 * (self.in_height // 64) * (self.in_width // 64)
        elif self.arc == 'celeba':
            in_channels = 512 * (self.in_height // 32) * (self.in_width // 32)
        else:
            raise ValueError('Invalid architecture {}'.format(self.arc))

        self.linear = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, 1024),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        if self.arc == 'places2':
            x = self.extra_conv_layer(x)
        return self.linear(x)


class ContextDiscriminator(nn.Module):
    def __init__(self, local_in_shape, global_in_shape, architecture='celeba'):
        super(ContextDiscriminator, self).__init__()
        self.arc = architecture
        self.input_shapes = [local_in_shape, global_in_shape]
        self.out_shape = (1,)
        self.local_D = LocalDiscriminator(local_in_shape)
        self.global_D = GlobalDiscriminator(global_in_shape)

        self.concat = Concatenate(-1)

        self.layers = nn.Sequential(
            nn.Linear(self.local_D.output_shape[-1] + self.global_D.out_shape[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_x, global_x = x
        local_x = self.local_D(local_x)
        global_x = self.global_D(global_x)
        to_ret = self.concat([local_x, global_x])
        return self.layers(to_ret)