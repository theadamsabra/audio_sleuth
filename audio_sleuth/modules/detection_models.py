import torch.nn as nn
import torch
from torch import Tensor
from audio_sleuth.modules.conv import Conv1dMaxPoolBlock 

class CQTFramewiseDetection(nn.Module):
    '''
    CQT framewise detection model.

    Args:

       conv_channels (list[int]): conv output channels.
       linear_channels (list[int]): fully connected layer channels.
       n_bins (int): number of CQT bins. default set to 84.
    '''
    def __init__(self, conv_channels:list[int], conv_kernels:list[int], conv_strides:list[int], \
                pool_kernels:list[int], pool_strides:list[int], n_bins:int=84, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels 
        self.conv_strides = conv_strides
        self.pool_kernels = pool_kernels
        self.pool_strides = pool_strides
        self.n_bins = n_bins

        # Construct all convs with pooling blocks given inputs
        self.convs = nn.ModuleList()
        for i in range(len(self.conv_channels)):
            if i == 0:
                in_channels = self.n_bins
                out_channels = self.conv_channels[i]
            else:
                in_channels = self.conv_channels[i-1]
                out_channels = self.conv_channels[i]

            self.convs.append(
                nn.Conv1d(
                    in_channels, out_channels, self.conv_kernels[i], self.conv_strides[i]
                ) 
            )
        self.linear = nn.Linear(736, 345)

    def forward(self, x:Tensor) -> Tensor:
        for layer in self.convs:
            x = torch.relu(layer(x))
        flattened_x = torch.flatten(x, 1, 2)
        output = torch.sigmoid(self.linear(flattened_x))
        return output
