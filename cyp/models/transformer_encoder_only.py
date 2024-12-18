from torch import nn
from torch.nn import functional as F
import torch

import numpy as np
import pandas as pd
import math
from pathlib import Path

from .base import ModelBase

class TransformerEncoderModel(ModelBase):
    """
    A transformer replacement of the RNN model. This model is based on the transformer
    
    Note that this class assumes feature_engineering was run with channels_first=True

    Parameters
    ----------
    in_channels: int, default=9
        Number of channels in the input data. Default taken from the number of bands in the
        MOD09A1 + the number of bands in the MYD11A2 datasets
    num_bins: int, default=32
        Number of bins in the histogram
    d_model: int, default=128
        The size of the hidden state. Default taken from the original repository
    nhead: int, default=8
        Number of heads in the transformer
    num_layers: int, default=6
        Number of transformer layers
    dim_feedforward: int, default=512
    dropout: float, default=0.1
        Dropout rate
    layer_norm_eps: float, default=1e-5
        Epsilon value for layer normalization
    activation: str, default=<function relu>
        Activation function to use
    savedir: pathlib Path, default=Path('data/models')
        The directory into which the models should be saved.
    device: torch.device
        Device to run model on. By default, checks for a GPU. If none exists, uses
        the CPU
    dtype: torch.dtype, default=torch.float32
        Data type to use for the model
    """

    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        d_model=128,
        nhead=8,
        num_encode_layers=6,
        dim_feedforward=512,
        layer_norm_eps=1e-5,
        dense_features=None,
        activation='relu',
        savedir=Path("data/models"),
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.01,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
    ):

        model = TransformerNet(
            in_channels=in_channels,
            num_bins=num_bins,
            d_model=d_model,
            nhead=nhead,
            num_encode_layers=num_encode_layers,
            dim_feedforward=dim_feedforward,
            dense_features=dense_features,
            layer_norm_eps=layer_norm_eps,
        )

        if dense_features is None:
            num_dense_layers = 2
        else:
            num_dense_layers = len(dense_features)
        model_weight = f"dense_layers.{num_dense_layers - 1}.weight"
        model_bias = f"dense_layers.{num_dense_layers - 1}.bias"



        super().__init__(
            model,
            model_weight,
            model_bias,
            "transformer_encoder",
            savedir,
            use_gp,
            sigma,
            r_loc,
            r_year,
            sigma_e,
            sigma_b,
            device,
        )

    def reinitialize_model(self, time=None):
        self.model.initialize_weights()

class TransformerNet(nn.Module):
    
    
    def __init__(self,
                 in_channels=9,
                 num_bins=32,
                 d_model=512,
                 nhead=8,
                 num_encode_layers=6,
                 dim_feedforward=2048,
                 dense_features=None,
                 layer_norm_eps=1e-5,
                 activation='relu'):
        super().__init__()
        
        if dense_features is None:
            dense_features = [1]
        dense_features.insert(0, d_model * num_bins)
        
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

        #turn on batch first
        self.encoder_layer.batch_first = True
        self.encoder_layer.self_attn.batch_first = True

        self.d_model = d_model

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encode_layers
        )

        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=dense_features[i - 1], out_features=dense_features[i]
                )
                for i in range(1, len(dense_features))
            ]
        )

        self.initialize_weights()
   
    def initialize_weights(self):
        """
        Initialize the weights of the transformer encoder and dense layers.
        
        This method initializes the weights of the transformer encoder using Xavier uniform initialization
        for weights and constant initialization for biases. It also initializes the weights of the dense
        layers using Kaiming uniform initialization for weights and constant initialization for biases.
        """
        for name, param in self.transformer_encoder.named_parameters():
            if "weight" in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight)
            nn.init.constant_(dense_layer.bias, 0)


    def forward(self, x, return_last_dense=False):
        """
        If return_last_dense is true, the feature vector generated by the second to last
        dense layer will also be returned. This is then used to train a Gaussian Process model.
        """
        # the model expects feature_engineer to have been run with channels_first=True, which means
        # the input is [batch, bands, times, bins].
        # Reshape to [batch, times, bands * bins]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])


        x = self.transformer_encoder(x, src_key_padding_mask=None)


        #convert to [batch, features]
        x = x.view(x.shape[0], -1)

        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
        #     if return_last_dense and (layer_number == len(self.dense_layers) - 2):
        #         output = x
        

        # if return_last_dense:
        #     return x, output
        # else:
        #     return x
        return x