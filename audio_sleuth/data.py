import os
import torch 
import librosa
import math
from torch.nn import Module
from torch import Tensor
from torch.utils.data import Dataset

class HalfTruthDataset(Dataset):
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
    def __init__(self, path_to_txt:str, fs:int, transform:Module=None) -> None:
        super().__init__()
        self.fs = fs
        self.transform = transform
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

        # TODO: Enable transforms
        # if self.transforms:
        #   audio, labels = self.transform(audio, labels)

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
