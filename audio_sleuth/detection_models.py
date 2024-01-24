import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

'''
Implementation from 
    https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
'''
class CNNLayerNorm(nn.Module):
    """
    Layer normalization for convolutional outputs. Co-opted from https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/

    Args:
        num_channels (int): number of channels of layer. 
    """
    def __init__(self, num_features:int):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x:Tensor) -> Tensor:
        '''
        Run layer normalization by permuting, running layer norm, and repermuting back
        to the original shape.

        Args:
            x (Tensor): input tensor of size (batch, channels, feature, time)
        
        Returns:
            normalized_x (Tensor): x normalized across the feature dimension. also of size (batch, channels, feature, time)
        '''
        # Permute time and feature dimensions
        x = x.transpose(2, 3).contiguous() 
        # Run layer norm on last dimension (feature)
        x = self.layer_norm(x)
        # Permute time and feature back to original when returning:
        return x.transpose(2, 3).contiguous() 

class DeepResidualNetworkWithLayerNorm(nn.Module):
    """
    Deep Residual network as propsed by He et al. Paper can be found [here](https://arxiv.org/pdf/1603.05027.pdf).
    We perform layer instead of batch normalization.

    Args:
        in_channels (int): input channels for first convolutional layer.
        out_channels (int): output channels of first and second convlutional layer. 
        kernel_size (int): kernel size of both convolutional layers.
        stride (int): stride of kernel over tensor.
        dropout (float): dropout percentage for dropout layers.
        num_features (int): number of channels for layer norm.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, dropout:float, num_features:int):
        super(DeepResidualNetworkWithLayerNorm, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(num_features)
        self.layer_norm2 = CNNLayerNorm(num_features)

    def forward(self, x:Tensor) -> Tensor:
        '''
        Core forward pass of the residual network. We run convolutions, dropouts, and gelu activations before
        adding the residual back to the output.

        Args:
            x (Tensor): input tensor of size (batch, channel, feature, time)
        
        Returns:
            residual_output (Tensor): output tensor of processed input + residual. output is of size (batch, channel, feature, time).
        '''
        residual = x 
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x 
        
class BidirectionalGRU(nn.Module):
    '''
    Wrapper of torch.nn.GRU to ensure it is bidirectional. Furthermore, we also add layer norms and dropout.

    Args:
        input_size (int): input size to RNN. 
        hidden_size (int): hidden size of RNN network
        dropout (float): dropout percentage for dropout layers
        batch_first (bool): flag to let torch.nn.GRU know if batch is the first dimension.
    '''
    def __init__(self, input_size:int, hidden_size:int, dropout:float, batch_first:bool):
        super(BidirectionalGRU, self).__init__()

        self.bidirectional_rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor) -> Tensor:
        '''
        Apply a bidirectional gated recurrent unit to input sequence, with additional layer norm,
        activation, and dropouts.

        Args:
            x (Tensor): input tensor of shape () if `batch_first` is True, and shape () otherwise.
        
        Returns:
            output (Tensor): output tensor of shape () if `batch_first` is False, and shape () otherwise.
        '''
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.bidirectional_rnn(x)
        x = self.dropout(x)
        return x

class DeepSpeech2(nn.Module):
    """
    Impelementation of Deep Speech 2 by Amodei et al. Paper can be found [here](https://arxiv.org/pdf/1512.02595.pdf).
    However, we leverage layer norm across the feature dimensions as opposed to the batch dimensions.

    Args:
        num_cnn_layers (int): number of residual convolutional layers. 
        num_rnn_layers (int): number of bidirectional gated recurrent unit layers.
        rnn_dimension (int): dimensionality of rnn.
        num_classes (int): number of classes to run prediction on.
        num_features (int): number of features for layer norm.
        stride (int): stride of convolutions.
        dropout (float): dropout percentage for dropout layers.
    """
    def __init__(self, num_cnn_layers:int, num_rnn_layers:int, rnn_dimension:int, \
                 num_classes:int, num_features:int, stride:int=2, dropout:float=0.1):
        super(DeepSpeech2, self).__init__()
        num_features = num_features//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            DeepResidualNetworkWithLayerNorm(in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            dropout=dropout,
                            num_features=num_features 
            )
            for _ in range(num_cnn_layers)
            ]
        )
        self.fully_connected = nn.Linear(num_features*16, rnn_dimension)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(input_size=rnn_dimension if i==0 else rnn_dimension*2,
                            hidden_size=rnn_dimension, 
                            dropout=dropout, 
                            batch_first=i==0 # batch first is true for the first layer
            )
            for i in range(num_rnn_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dimension*2, rnn_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dimension, num_classes)
        )

    def forward(self, x:Tensor) -> Tensor:
        '''
        Classify using Deep Speech 2.
        
        Args:
            x (Tensor): input tensor of shape (batch_size, num)
        
        Return:
            classfication_output (Tensor): output tensor of shape (batch_size, time, num_classes)
        '''
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], int(sizes[1] * sizes[2] / 2), int(sizes[3]*2))  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x.permute(0,2,1).float()