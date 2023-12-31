import os
import random
import torch 
import librosa
import math
from torch.nn import Module
from torch import Tensor
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    '''
    Base class for all datasets. Contains general functions leveraged by all other classes.
    
    Args:
        duration_sec (float): duration of crop in seconds. if set to None, the whole file
        will be processed.
        fs (int): sampling rate of file.
        transform (Module): audio augmentation pipeline. default set to None.
    '''
    def __init__(self, duration_sec:float, fs:int, transform:Module=None) -> None:
        super().__init__()
        self.duration_sec = duration_sec
        
        if self.duration_sec:
            self.process_whole_file = False
        else:
            self.process_whole_file = True

        self.fs = fs
        self.transform = transform

    def __len__(self):
        '''Will be overwritten for each dataset.'''
        pass 

    def __getitem__(self, idx):
        '''Will be overwritten for each dataset.'''
        pass

    def _construct_random_indices(self, vector:Tensor) -> tuple[int, int]:
        '''
        Construct random indices from length of vector.

        Args:
            vector (Tensor): samplewise labels of real/fake speech. 
        
        Returns:
            start_idx (int): start index of vector.
            end_idx (int): end index of vector.
        '''
        duration_samples = int(self.duration_sec * self.fs)

        start_idx = random.randrange(0, len(vector)-duration_samples)
        end_idx = start_idx + duration_samples

        return start_idx, end_idx 


class HalfTruthDataset(BaseDataset):
    '''
    Torch dataset of Half Truth Dataset by Jiangyan Yi, Ye Bai, Jianhua Tao, Haoxin Ma, Zhengkun Tian, 
    Chenglong Wang, Tao Wang, and Ruibo Fu.
    
    Data can be downloaded here: https://zenodo.org/records/10377492
    Paper can be read here: https://arxiv.org/pdf/2104.03617.pdf

    Args:
        path_to_txt (str): path to text file containing paths and ground truth labels. assumes absolute path for easier parsing.
        duration_sec (float): duration of crop in seconds.
        fs (int): sampling rate of file.
        transform (Module): audio augmentation pipeline. default set to None.
    '''
    def __init__(self, path_to_txt:str, duration_sec:float, fs:int, transform:Module=None) -> None:
        super().__init__(duration_sec, fs, transform)
        self.path_to_txt = path_to_txt
        self.text_file = open(self.path_to_txt, 'r').read()
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

        # Crop audio and samplewise labels if needed:
        if not self.process_whole_file:
            # Generate start and end indices to crop:
            start_idx, end_idx = self._construct_random_indices(audio)
            audio = audio[start_idx:end_idx]
            labels = labels[start_idx:end_idx]

        # # Transform audio if needed:
        # if self.transform:
        #     audio, labels = self.transform(audio, labels)

        return audio, labels 
     
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
