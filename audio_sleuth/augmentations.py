import torch
import math
import numpy as np
import torch.nn as nn
import torchaudio.transforms as T
from torch import Tensor

class LabelAlignment(nn.Module):
    '''
    Label Alignment class to help with aligning sample wise labels from the time domain into other representations.

    Args:
        win_size (int): window size of transformations.
        hop_size (int): hop size of transformations.
        pad_samples (int): number of samples to pad vector. default set to 0.
        pad_mode (str): type of pad mode. default set to `"constant"`. supported pad modes can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html).
        pad_value (int): value to pad with. default set to 0 (i.e. zero padding.)
    '''
    def __init__(self, win_size:int, hop_size:int, pad_mode:str='constant', pad_value:int=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.win_size = win_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def _pad_vector(self, vector:Tensor) -> Tensor:
        '''
        Zero pad vector on both sides to align with torchaudio based FFT.

        Args:
            vector (Tensor): arbitrary 1D vector. 
        
        Returns:
            padded_vector (Tensor): padded 1D vector.
        '''
        # Zero pad if necessary:
        signal_dim = vector.dim()
        extended_shape = [1] * (3-signal_dim) + list(vector.size())
        pad = int(self.win_size // 2)
        padded_vector = torch.nn.functional.pad(vector.view(extended_shape), [pad, pad], 
                                             mode=self.pad_mode, value=self.pad_value)
        return padded_vector.flatten()

    def forward(self, vector:Tensor) -> Tensor:
        '''
        Align labels given hop size and window size.

        Args:
            vector (Tensor): arbitrary 1D vector. 
        
        Returns:
            aligned_vector (Tensor): padded and framed tensor.
        '''
        padded_vector = self._pad_vector(vector)
        framed_vector = padded_vector.unfold(-1, self.win_size, self.hop_size)
        return framed_vector.mean(dim=-1)

class Resample(nn.Module):
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

        self.resample_to_new = T.Resample(self.input_sr, self.new_sr)  
        self.resample_to_original = T.Resample(self.new_sr, self.input_sr)

    def forward(self, audio:Tensor, labels:Tensor) -> tuple[Tensor, Tensor]:
        '''
        Core implementation of resample block.

        Args:
            audio (Tensor): audio tensor in time domain.
            labels (Tensor): samplewise labels in time domain. 

        Returns:
            resampled_audio (Tensor): resampled audio tensor.
            resampled_labels (Tensor): resampled label tensor.
        '''
        audio = self.resample_to_new(audio) 
        resampled_labels = self.resample_to_new(labels)

        if self.return_original_sr:
            return self.resample_to_original(audio), labels
        else:
            return audio, resampled_labels

class DynamicResampler(nn.Module):
    '''
    Dynamic resampler to randomly resample to a specific sampling rate with specific weights. 
    If weights are not specified, uniform weighting will be applied. All audio vectors will be resampled back into 
    `input_sr` for uniform shapes after other augmentations. The goal is to allow for the network to generalize classifications
    across varying bandwidths.

    Args:
        input_sr (int): input sampling rate of audios.
        output_sr (list[int]): list of sampling rates to be randomly selected from.
        output_sr_weights (list[float]): optional param for weights of each sampling rate. 
        will use uniform sampling rate if not specified.
    '''
    def __init__(self, input_sr:int, output_sr:list[int], output_sr_weights:list[float]=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_sr = input_sr
        self.output_sr = output_sr
        # If weights are not specified, just use uniform sampling
        if output_sr_weights:
            self.output_sr_weights = output_sr_weights
        else:
            self.output_sr_weights = [1 / (len(self.output_sr))] * len(self.output_sr)

    def _select_rate(self):
        '''Randomly select sampling rate.'''
        return np.random.choice(a=self.output_sr, p=self.output_sr_weights)

    def forward(self, audio:Tensor, labels:Tensor) -> tuple[Tensor, Tensor]:
        '''
        Dynamically select sampling rate and resample to/from old and new sampling rate.

        Args:
            audio (Tensor): audio tensor in time domain.
            labels (Tensor): samplewise labels in time domain. 

        Returns:
            resampled_waveform (Tensor): resampled audio tensor.
            resampled_labels (Tensor): resampled label tensor.
        '''
        selected_rate = self._select_rate()
        resampler = Resample(self.input_sr, selected_rate)
        resampled_audio, resampled_labels = resampler(audio, labels)
        return resampled_audio, resampled_labels 

class LFCC(nn.Module):
    '''
    Linear Frequency Cepstral Coefficient augmentation. We leverage torchaudio's implementation, however, we ensure the
    samplewise labels from the dataset are also aligned.

    Args:
        fs (int): sampling rate of audio.
        n_fft (int): number of FFT - creates n_fft // 2 + 1 bins.
        hop_size (int): hop size of transformation.
        win_size (int): window size of transformation.
        n_filters (int): number of linear filters.
        n_lfcc (int): number of linear frequency cepstral coefficients.
        center (bool): flag to pad audio file. default set to True.
    '''
    def __init__(self, fs:int, n_fft:int, hop_size:int, win_size:int, n_filters:int, \
                 n_lfcc:int, center:bool=True, speckwargs:dict=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.n_filters = n_filters
        self.n_lfcc = n_lfcc
        self.center = center
        if speckwargs:
            self.speckwargs = speckwargs
        else:
            self.speckwargs = {
                'n_fft': self.n_fft,
                'hop_length': self.hop_size,
                'win_length': self.win_size,
                'center': self.center
            }
        self.lfcc_extractor = T.LFCC(
            sample_rate=self.fs,
            n_filter=self.n_filters,
            n_lfcc=self.n_lfcc,
            speckwargs=self.speckwargs
        ) 
        self.label_aligner = LabelAlignment(self.hop_size, self.win_size)

    def forward(self, audio:Tensor, labels:Tensor) -> tuple[Tensor, Tensor]:
        '''
        Convert audio into LFCC and frame labels.

        Args:
            audio (Tensor): audio tensor in time domain.
            labels (Tensor): samplewise labels in time domain. 
        
        Returns:
            lfcc_tensor (Tensor): lfcc representation of audio of shape (n_lfcc, win_size)
            framed_labels (Tensor): framed and averaged labels of shape (num_frames, win_size)
        '''
        lfcc_tensor = self.lfcc_extractor(audio)
        framed_labels = self.label_aligner(labels)
        return lfcc_tensor, framed_labels

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
    
    def forward(self, audio:Tensor, labels:Tensor) -> tuple[Tensor, Tensor]:
        '''
        Loop through each augmentation and sequentially run the forward.

        Args:
            audio (Tensor): audio tensor to be transformed.
            labels (Tensor): label tensor to be transformed.
        
        Returns:
            audio (Tensor): transformed audio tensor.
            labels (Tensor): transformed label tensor.
        '''
        for augmentation in self.augmentation_list:
            audio, labels = augmentation(audio, labels)  

        return audio, labels
