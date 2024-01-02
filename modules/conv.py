from torch import Tensor
from torch import nn


class Conv1dMaxPoolBlock(nn.Module):
    '''
    Convolutional Block with Max Pooling.

    Args:
        channel_in (int): in channels of convolutions
        channel_out (int): out channels of convolutions
        conv_kernel (int): kernel size
        conv_stride (int): stride of convolution
        pool_kernel (int): kernel of max pool
        pool_stride (int): stride of max pool.
    '''
    def __init__(self, channel_in:int, channel_out:int, \
                 conv_kernel:int, conv_stride:int, \
                    pool_kernel:int, pool_stride:int,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(channel_in, channel_out, conv_kernel, conv_stride)
        self.pool = nn.MaxPool1d(pool_kernel, pool_stride)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.pool(self.conv(x))