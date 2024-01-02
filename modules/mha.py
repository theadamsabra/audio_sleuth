import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    '''
    Core implementation of scaled dot product attention.
    
    Args:
        temp (float): temperature of attention mechanism.
    '''
    def __init__(self, temp:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temp = temp

    def forward(self, Q:Tensor, K:Tensor, V:Tensor) -> Tensor:
        '''
        Implement basic attention with queries, keys, and values.

        Args:
            Q (Tensor): query matrix of size [batch_size, ]
            K (Tensor): key matrix of size [batch_size, ]
            V (Tensor): value matrix of size [batch_size, ]
        
        Returns:
            attention (Tensor): final attention output of size [batch_size, ]
        '''
        # "Transpose" K
        K = K.permute(0, 2, 1) 
        # Get intermediate softmax output
        softmax_ = F.softmax(
            (torch.dot(Q, K) / self.temp)
        )
        # Get final attention output
        attention = torch.dot(softmax_, V)
        return attention
        

class MultiHeadAttention(nn.Module):
    '''
    Simple implementation of multihead attention.

    Args:

    '''
    def __init__(self, n_heads, embedding_dimension, temp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads 
        self.embedding_dimension = embedding_dimension
        self.temp = temp
        self.scaled_dpa = ScaledDotProductAttention(self.temp)
    
    def forward(self, x:Tensor) -> Tensor:
        pass 