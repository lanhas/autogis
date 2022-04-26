import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import register


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, input_channels, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(input_channels)),
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(input_channels, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)

        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(input_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(input_channels, output_channels,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i+1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseUnet(nn.Module):
    def __init__(self):
        super(DenseUnet, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.db1 = _DenseBlock(3, 64)
        self.trans1 = _Transition(256, 128)

        self.db2 = _DenseBlock(3, 128)
        self.trans2 = _Transition(512, 256)

        self.db3 = _DenseBlock(3, 256)
        self.trans3 = _Transition(1024, 512)

        self.db4 = _DenseBlock(3, 512)

        self.bn = nn.BatchNorm2d(1024)
        self.center = nn.Sequential(nn.MaxPool2d(),
                                    nn.Conv2d(),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(),
                                    nn.ReLU())

        self.up5 = self.upsample(256, 256)
        self.up4 = self.upsample(256, 128)
        self.up3 = self.upsample(128, 128)
        self.up2 = self.upsample(128, 64)
        self.up1 = self.upsample(64, 32)

        self.conv5 = self.conv_stage()
        self.conv4 = self.conv_stage()
        self.conv3 = self.conv_stage()
        self.conv2 = self.conv_stage()
        self.conv1 = self.conv_stage()

        self.conv_last = nn.Sequential(nn.Conv2d(32, 1),
                                       nn.Sigmoid())

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        # encoding
        conv_first = self.conv_first(input)

        db1 = self.db1(conv_first)
        trans1 = self.trans1(db1)

        db2 = self.db1(trans1)
        trans2 = self.trans2(db2)

        db3 = self.db3(trans2)
        trans3 = self.trans3(db3)

        db4 = self.db4(trans3)
        bn = self.bn(db4)

        # center
        center = self.center(bn)

        # decoding
        decode5 = self.decode5(bn, center)

        decode4 = self.decode4(db3, decode5)

        decode3 = self.decode3(db2, decode4)

        decode2 = self.decode2(db1, decode3)

        decode1 = self.decode1(conv_first, decode2)
        out = self.conv_last(decode1)

        return out







@register('dense-unet')
def dense_unet():
    """Constructs an dense_unt model with embedding Network"""
    return DenseUnet()
