import os
import torch 
import librosa
import math
from torch import Tensor
from torch.utils.data import Dataset

'''
Helper functions
'''
def find_all_wav_files(dir_:str):
    '''Find all wav files in directory'''
    files = []
    
    # Walk across all dir/subdirs
    for root, _, filenames in os.walk(dir_):
        # Get the full paths of the wav if it is in this dir and keep it
        tmp = [os.path.join(root, filename) for filename in filenames if '.wav' in filename]
        files += tmp
    return files

'''
Datasets
'''
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
    def __init__(self, path_to_txt:str, fs:int) -> None:
        super().__init__()
        self.fs = fs
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

class WaveFake(Dataset):
    '''
    Torch dataset of WaveFake dataset introduced by Joel Frank and Lea Schönherr.

    Args:
        real_root_dir (str): root dir of real dataset(s).
        generated_root_dir (str): root directory of wavefake dataset (aka generated data.)
        fs (int): sampling rate of file.
        transform (Module): audio augmentation pipeline. default set to None.
    '''
    def __init__(self, real_root_dir:str, generated_root_dir:str, fs:int) -> None:
        super().__init__()
        self.real_root_dir = real_root_dir
        self.generated_root_dir = generated_root_dir
        self.fs = fs
        # Parse out relevant information:
        self.real_root_dir_wavs = find_all_wav_files(self.real_root_dir)
        self.generated_root_dir_wavs = find_all_wav_files(self.generated_root_dir)
        self.all_files = self.real_root_dir_wavs + self.generated_root_dir_wavs

    def __len__(self): 
        return len(self.all_files)

    def __getitem__(self, idx): 
        # Get filepath:
        filepath = self.all_files[idx]

        # Check if which root dir is in the path.
        # This will serve as our binary labels.
        label = 0 if self.real_root_dir in filepath else 1

        # Load in audio, and construct sample-wise labels:
        audio, _ = librosa.load(filepath, sr=self.fs)
        audio = torch.from_numpy(audio)
        labels = torch.full(audio.shape, label)

        return audio, labels