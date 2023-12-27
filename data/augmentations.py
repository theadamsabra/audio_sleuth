import torch
import torch.nn as nn
from torchaudio.transforms import Resample

class ResampleBlock(nn.Module):
    '''
    Resample block for resampling audio. Generally used for resampling down -> back to original sampling rate as augmentation.
    However, specifying `return_original_sr` to False allows for a simple down/up sampling.

    Args:
        input_sr (int): sampling rate of input audio file.
        new_sr (int): new sampling rate.
        return_original_sr (bool): return file to `input_sr`. default set to True.
    '''
    def __init__(self, input_sr:int, new_sr:int, return_original_sr:bool=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_sr = input_sr
        self.new_sr = new_sr
        self.return_original_sr = return_original_sr

        self.resample_to_new = Resample(self.input_sr, self.new_sr)  
        self.resample_to_original = Resample(self.new_sr, self.input_sr)

    def forward(self, waveform:torch.Tensor) -> torch.Tensor:
        '''
        Core implementation of resample block.

        Args:
            waveform (torch.Tensor): audio tensor in time domain.
        
        Returns:
            resampled_waveform (torch.Tensor): resampled audio tensor.
        '''
        waveform = self.resample_to_new(waveform) 
        if self.return_original_sr:
            return self.resample_to_original(waveform)
        else:
            return waveform