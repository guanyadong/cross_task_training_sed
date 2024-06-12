import warnings

import torch.nn as nn
import torch
from .RNN import BidirectionalGRU, BiRNN
from .CNN import CNN, Densenet


class CRNN(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        nclass=10,
        attention=True,
        activation="glu",
        dropout=0.5,
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        freeze_bn=False,
        use_embeddings=False,
        embedding_size=527,
        embedding_type="global",
        frame_emb_enc_dim=512,
        **kwargs,
    ):
        """
            Initialization of CRNN model
        
        Args:
            n_in_channel: int, number of input channel
            n_class: int, number of classes
            attention: bool, adding attention layer or not
            activation: str, activation function
            dropout: float, dropout
            train_cnn: bool, training cnn layers
            rnn_type: str, rnn type
            n_RNN_cell: int, RNN nodes
            n_layer_RNN: int, number of RNN layers
            dropout_recurrent: float, recurrent layers dropout
            freeze_bn: 
            **kwargs: keywords arguments for CNN.
        """
        super(CRNN, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.freeze_bn = freeze_bn
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type

        n_in_cnn = n_in_channel

        self.cnn_0_1 = CNN(n_in_channel=n_in_cnn, activation=activation, conv_dropout=0.2,
                       kernel_size=[3, 3],
                       padding=[1, 1], stride=2 * [1], nb_filters=[32, 64],
                       pooling=[[2, 2], [2, 2]],
                       normalization="subspectral",
                       )

        self.cnn_2 = CNN(n_in_channel=64, activation=activation, conv_dropout=0.2,
                       kernel_size=[3],
                       padding=[1], stride=[1], nb_filters=[128],
                       pooling=[[1, 2]],
                       normalization="subspectral",
                       )

        self.cnn_3 = CNN(n_in_channel=128, activation=activation, conv_dropout=0.2,
                       kernel_size=[3],
                       padding=[1], stride=[1], nb_filters=[256],
                       pooling=[[1, 2]],
                       normalization="subspectral",
                       )

        self.cnn_4 = CNN(n_in_channel=256, activation=activation, conv_dropout=0.2,
                       kernel_size=[3],
                       padding=[1], stride=[1], nb_filters=[256],
                       pooling=[[1, 2]],
                       normalization="batch",
                       )

        self.cnn_sed = CNN(n_in_channel=256, activation=activation, conv_dropout=0.2,
                        kernel_size=[3, 3],
                        padding=[1, 1], stride=2 * [1], nb_filters=[256, 256],
                        pooling=[[1, 2], [1, 2]])

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if rnn_type == "BGRU":
            nb_in = 256

            self.rnn_sed = BidirectionalGRU(
                n_in=256,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )

            self.rnn_vad = BidirectionalGRU(
                n_in=256,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )

        self.densenet = Densenet(channel=384)
        self.densenet2 = Densenet(channel=384)

        self.dropout = nn.Dropout(0.5)
        self.dropout_vad = nn.Dropout(0.5)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.dense_vad = nn.Linear(n_RNN_cell*2, 3)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
            self.softmax = nn.Softmax(dim=-1)


        if self.use_embeddings:
            if self.embedding_type == "frame":
                self.frame_embs_encoder = nn.GRU(batch_first=True, input_size=embedding_size,
                                                      hidden_size=512,
                                                      bidirectional=True)
                self.shrink_emb = torch.nn.Sequential(torch.nn.Linear(2 * frame_emb_enc_dim, nb_in),
                                                      torch.nn.LayerNorm(nb_in))
            else:
                self.shrink_emb = torch.nn.Sequential(torch.nn.Linear(embedding_size, nb_in),
                                                      torch.nn.LayerNorm(nb_in))
            self.cat_tf = torch.nn.Linear(2*nb_in, nb_in)

        self.pooling4 = nn.AvgPool2d([1, 4])
        self.pooling2 = nn.AvgPool2d([1, 2])

        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, pad_mask=None, embeddings=None):

        x = x.transpose(1, 2).unsqueeze(1)

        # input size : (batch_size, n_channels, n_frames, n_freq)

        # conv features

        x_0_1 = self.cnn_0_1(x)
        x_2 = self.cnn_2(x_0_1)  # add      64, 156, 16
        x_3 = self.cnn_3(x_2)    # add      128, 156, 8
        x_4 = self.cnn_4(x_3)    # add      128, 156, 4

        x_new_2 = self.pooling4(x_2)
        x_new_3 = self.pooling2(x_3)
        x_cat = torch.cat([x_new_3, x_new_2], dim=1)
        x_cat1 = self.densenet(x_cat)

        x_cat1 = x_cat1 * self.alpha + x_4

        x_cat2 = self.densenet2(x_cat)
        x_cat2 = x_cat2 * self.beta + x_4

        # vad cnn features
        x_sed = self.cnn_sed(x_cat2)
        x_vad = torch.mean(x_cat1, dim=-1)   # [bs, chan, frames]

        x_sed = x_sed.squeeze(-1)
        x_sed = x_sed.permute(0, 2, 1)  # [bs, frames, chan]

        x_vad = x_vad.squeeze(-1)
        x_vad = x_vad.permute(0, 2, 1)  # [bs, frames, chan]

        x_vad = self.rnn_vad(x_vad)
        x_vad = self.dropout_vad(x_vad)
        strong_vad = self.dense_vad(x_vad)  # [bs, frames, 3]

        x_sed = self.rnn_sed(x_sed)
        x_sed = self.dropout(x_sed)
        strong = self.dense(x_sed)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)

        if self.attention:
            sof = self.dense_softmax(x_sed)  # [bs, frames, nclass]
            if not pad_mask is None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong.transpose(1, 2), weak, strong_vad

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
