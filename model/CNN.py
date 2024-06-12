import torch.nn as nn
import torch

class Densenet(nn.Module):
    def __init__(self, channel=None):
        super(Densenet, self).__init__()
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(channel, eps=0.001, momentum=0.99)

        self.conv = nn.Conv2d(in_channels=channel, out_channels=256, kernel_size=1, padding=0, stride=1)

        self.batchnorm2 = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv(x)

        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x



class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class SubSpectralNorm(nn.Module):
    def __init__(self, channel, group):
        super(SubSpectralNorm, self).__init__()
        self.group = group
        self.channel = channel
        self.norm = nn.BatchNorm2d(self.channel * group, eps=0.001, momentum=0.99)

    def forward(self, x):  # B,C,T,F
        B, C, T, F = x.size()
        x = x.permute(0, 1, 3, 2)    # B,C,F,T
        x = x.contiguous().view(B, C * self.group, F // self.group, T)   # B,C*group,F//group,T
        x = self.norm(x)                                                 # B,C*group,F//group,T
        x = x.view(B, C, F, T)     # B,C,F,T
        x = x.permute(0, 1, 3, 2)  # B,C,T,F

        return x


class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="Relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        **transformer_kwargs
    ):
        """
            Initialization of CNN network s
        
        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
            )
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )

            elif normalization == "subspectral":
                cnn.add_module(
                    "subspectralnorm{0}".format(i),
                    SubSpectralNorm(channel=nOut, group=4),
                )

            # elif normalization == "layer":
            #     cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels

        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.cnn(x)
        return x
