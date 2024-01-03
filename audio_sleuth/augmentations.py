import math
import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.transforms import Resample

class LabelAlignment(nn.Module):
    '''
    Label Alignment class to help with aligning sample wise labels from the time domain into other representations.

    Args:
        hop_size (int): hop size of transformations.
        win_size (int): window size of transformations.
    '''
    def __init__(self, hop_size:int, win_size:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hop_size = hop_size
        self.win_size = win_size

    def _pad_vector(self, vector:Tensor) -> Tensor:
        '''
        Pad vector for framing. Currently only supporting reflection padding.
        
        Args:
            vector (Tensor): arbitrary 1D vector. 
        
        Returns:
            padded_vector (Tensor): padded vector to accomodate framing.
        '''
        # Calculate total len needed with padding and get differnce
        total_len = math.ceil(len(vector) / self.hop_size) * self.hop_size
        pad_len = total_len - len(vector)

        if pad_len % 2 == 0:
            right = left = int(pad_len / 2)
        else:
            right = int(pad_len / 2)
            left = int(pad_len / 2) + (pad_len % 2)

        # Pad through reflection
        left_pad_label = vector[0].item()
        right_pad_label = vector[-1].item()

        left_padding = Tensor([left_pad_label] * left)
        right_padding = Tensor([right_pad_label] * right)

        return torch.cat([left_padding, vector, right_padding])

    def _frame_vector(self, vector:Tensor) -> Tensor:
        '''
        Frame vector of samplewise labels by win length and hop size of FFT. Take mean of every frame to
        generate frame-wise labels. Framing is done after padding.

        Args:
            vector (Tensor): arbitrary 1D vector. 
        
        Returns:
            framed_vector (Tensor): fake speech probability of frame.
        '''
        framed_labels = vector.unfold(0, self.win_size, self.hop_size)
        return torch.mean(framed_labels, dim=-1)

class ResampleBlock(nn.Module):
    '''
    Resample block for resampling audio. Generally used for resampling down -> back to original sampling rate as augmentation.

    Args:
        input_sr (int): sampling rate of input audio file.
        new_sr (int): new sampling rate.
        return_original_sr (bool): flag to determine if we resample back to the original sampling rate. default set to True.
    '''
    def __init__(self, input_sr:int, new_sr:int, return_original_sr:bool=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_sr = input_sr
        self.new_sr = new_sr
        self.return_original_sr = return_original_sr 

        self.resample_to_new = Resample(self.input_sr, self.new_sr)  
        self.resample_to_original = Resample(self.new_sr, self.input_sr)

    def forward(self, waveform:Tensor) -> Tensor:
        '''
        Core implementation of resample block.

        Args:
            waveform (Tensor): audio tensor in time domain.
        
        Returns:
            resampled_waveform (Tensor): resampled audio tensor.
        '''
        waveform = self.resample_to_new(waveform) 
        if self.return_original_sr:
            return self.resample_to_original(waveform)
        else:
            return waveform

class Augmentations(nn.Module):
    '''
    Core augmentation class to allow for a chain of augmentations applied to data. Used as core transformation internally.

    Args:
        augmentation_list (list[nn.Module]): list of all augmentations to be applied to the data. each augmentation
        must be inherited from nn.Module.
    '''
    def __init__(self, augmentation_list:list[nn.Module], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augmentation_list = nn.ModuleList(augmentation_list)
    
    def forward(self, x:Tensor) -> Tensor:
        '''Loop through each augmentation and sequentially run the forward.'''
        for augmentation in self.augmentation_list:
            x = augmentation(x)  
        return x