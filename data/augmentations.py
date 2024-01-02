import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.transforms import Resample

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