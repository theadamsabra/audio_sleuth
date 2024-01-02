import os
import random
import torch 
import librosa
import math
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset

class HalfTruthDataset(Dataset):
    '''
    Torch dataset of Half Truth Dataset by Jiangyan Yi, Ye Bai, Jianhua Tao, Haoxin Ma, Zhengkun Tian, 
    Chenglong Wang, Tao Wang, and Ruibo Fu.
    
    Data can be downloaded here: https://zenodo.org/records/10377492
    Paper can be read here: https://arxiv.org/pdf/2104.03617.pdf

    Args:
        path_to_txt (str): path to text file containing paths and ground truth labels. assumes absolute path for easier 
        parsing.
        duration_sec (int): duration of crop in seconds.
        fs (int): sampling rate of file. default to 44100.
        transform (nn.Module): audio augmentation pipeline. default set to None.
    '''
    def __init__(self, path_to_txt:str, duration_sec:int, fs:int=44100, \
                 transform:nn.Module=None, hop_size:int=128, win_size:int=128, *args, **kwargs) -> None:
        super().__init__()
        self.__dict__.update(kwargs)
        # Get path and open text file:
        self.path_to_txt = path_to_txt
        self.text_file = open(self.path_to_txt, 'r').read()
        # Store necessary params:
        self.duration_sec = duration_sec
        self.fs = fs
        self.transform = transform
        self.hop_size = hop_size
        self.win_size = win_size
        # Construct additional params from path and metadata:
        self.root_dir = os.path.dirname(self.path_to_txt)
        self.set_type = os.path.basename(self.root_dir).split('_')[-1]
        self.observations = self.text_file.split('\n')

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # Load in observation and get relevant information.
        observation = self.observations[idx] 
        filename, timestamps, _ = observation.split(' ')
        
        # Load file and construct to torch tensor
        audio, _ = librosa.load(os.path.join(self.root_dir, self.set_type, f'{filename}.wav'), sr=self.fs)
        audio = torch.from_numpy(audio)
        num_samples_audio = len(audio)

        # Map timestamps to samplewise labels.
        labels = self._generate_timestamps(timestamps, num_samples_audio)

        # Crop audio and samplewise labels  
        start_idx, end_idx, audio = self._fit_duration(audio)
        _, _, labels = self._fit_duration(labels, start_idx, end_idx)

        if self.transform:
            # Transform audio:
            audio = self.transform(audio)
            # Pad labels on both sides to accomodate spectrogram: 
            padded_labels = self._pad_labels(labels) 
            labels = self._frame_labels(padded_labels)

        return audio, labels 

    def _pad_labels(self, labels:Tensor) -> Tensor:
        '''
        Pad ground truth labels through reflection padding.
        
        Args:
            labels (Tensor): samplewise labels of real/fake speech. 
        
        Returns:
            padded_labels (Tensor): padded vector to accomodate framing.
        '''
        # Calculate total len needed with padding and get differnce
        total_len = math.ceil(len(labels) / self.hop_size) * self.hop_size
        pad_len = total_len - len(labels)
        pad_len_both_sides = pad_len // 2

        # Pad through reflection
        left_pad_label = labels[0].item()
        right_pad_label = labels[-1].item()

        left_padding = Tensor([left_pad_label] * pad_len_both_sides)
        right_padding = Tensor([right_pad_label] * pad_len_both_sides)

        return torch.cat([left_padding, labels, right_padding])
        

    def _frame_labels(self, labels:Tensor) -> Tensor:
        '''
        Frame vector of samplewise labels by win length and hop size of FFT. Take mean of every frame to
        generate frame-wise labels. Framing is done after padding.

        Args:
            labels (Tensor): samplewise labels of real/fake speech. 
        
        Returns:
            framed_labels (Tensor): fake speech probability of frame.
        '''
        framed_labels = labels.unfold(0, self.win_size, self.hop_size)
        return torch.mean(framed_labels, dim=-1)

    def _fit_duration(self, vector:Tensor, \
                      start_idx:int=None, end_idx:int=None) -> tuple[int, int, Tensor]:
        '''
        Fit specfied duration. AKA crop vector and return crop indices.

        Args:
            vector (Tensor): vector to be cropped.
            start_idx (int): start index of crop. default set to None.
            end_idx (int): end index of crop. default set to None.

        Returns:
            start_idx (int): start index of crop.
            end_index (int): end index of crop.
            vector (Tensor): cropped vector of len (end_index - start_index).
        '''
        duration_samples = int(self.duration_sec * self.fs)

        # Get start/end idx if not defined
        if (not start_idx) and (not end_idx):
            start_idx = random.randrange(0, len(vector)-duration_samples)
            end_idx = start_idx + duration_samples

        return start_idx, end_idx, vector[start_idx:end_idx]


    def _generate_timestamps(self, timestamps:str, num_samples_audio:int) -> Tensor:
        '''
        Helper function to generate array of timestamp labels.

        Args:
            timestamps (str): timestamps of real/fake data in audio. example is something like '0.00-4.96-T/4.96-5.65-F/5.65-9.26-T'.
            num_samples_audio (int): number of samples in audio. used to align last samples of timestamps to audio. 

        Returns:
            samplewise_labels (Tensor): samplewise labels of data.
        '''
        # Split by / 
        split_timestamps = timestamps.split('/')
        samplewise_labels = []

        # Construct all ground truth labels from timestamp
        for ts in split_timestamps:
            start, end, label = ts.split('-')
            # Convert from string to float
            start, end = float(start), float(end) 
            # Convert T/F to 0/1 respectively.
            label = 0 if label == 'T' else 1
            # Calculate number of samples and add it to the labels
            num_samples = (end*self.fs) - (start*self.fs) 
            labels = [label] * math.ceil(num_samples)
            samplewise_labels += labels

        # Sanity check for labels as timestamps only have 3 significant figures.
        diff = num_samples_audio - len(samplewise_labels)
        # If greater, we add the additional labels
        if diff > 0:
            diff_labels = [label] * diff
            samplewise_labels += diff_labels
        # Otherwise, we remove the extra labels
        elif diff < 0:
            samplewise_labels = samplewise_labels[0:diff]
        
        return Tensor(samplewise_labels)
