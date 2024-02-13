import os
import torch 
import librosa
import math
import subprocess
import zipfile
from torch import Tensor
from torch.utils.data import Dataset

'''
Helper functions
'''
def find_all_wav_files(dir_:str) -> list:
    '''Find all wav files in directory'''
    files = []
    
    # Walk across all dir/subdirs
    for root, _, filenames in os.walk(dir_):
        # Get the full paths of the wav if it is in this dir and keep it
        tmp = [os.path.join(root, filename) for filename in filenames if '.wav' in filename]
        files += tmp
    return files

def download_halftruth_dataset(save_root:str, remove_zip:bool=True):
    '''
    Downloader for the Half-Truth dataset from Zenodo by Jiangyan Yi et al.

    Link to dataset: https://zenodo.org/records/10377492
    Link to Half-Truth paper: https://arxiv.org/abs/2104.03617 
    
    Args:
        save_root (str): root directory to save data in.
        remove_zip (bool): flag to delete zip file or not. default set to True.
    '''
    link_addr = 'https://zenodo.org/records/10377492/files/HAD.zip'

    # Check if directory exists, if not make it:
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    
    path_to_zip = os.path.join(save_root, 'HAD.zip')

    print('Downloading Half Truth Dataset:')
    subprocess.run(['curl', link_addr, '--output', path_to_zip])
    print('Completed.')

    print('Extracting zip file:')
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(save_root)
    print('Completed.') 

    if remove_zip:
        os.remove(path_to_zip)

def download_wavefake(save_root:dir, remove_zip:bool):
    '''
    Downloader for the Wave Fake dataset from Zenodo by Joel Frank and Lea Schonherr.
    
    Link to dataset: https://zenodo.org/records/5642694
    Link to Wave Fake paper: https://arxiv.org/abs/2111.02813
    
    Args:
        save_root (str): root directory to save data in.
        remove_zip (bool): flag to delete zip file or not. default set to True.
    '''
    link_addr = 'https://zenodo.org/records/5642694/files/generated_audio.zip'

    # Check if directory exists, if not make it:
    if not os.path.isdir(save_root):
        os.mkdir(save_root)

    wavefake_dir = os.path.join(save_root, 'wavefake')
    if not os.path.isdir(wavefake_dir):
        os.mkdir(wavefake_dir)

    path_to_zip = os.path.join(save_root, 'wavefake.zip')

    print('Downloading Wave Fake Dataset:')
    subprocess.run(['curl', link_addr, '--output', path_to_zip])
    print('Completed.')

    print('Extracting zip file:')
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(wavefake_dir)
    print('Completed.') 

    if remove_zip:
        os.remove(path_to_zip)

'''
Core dataset classes:
'''

class HalfTruthDataset(Dataset):
    '''
    Torch dataset of Half Truth Dataset by Jiangyan Yi, Ye Bai, Jianhua Tao, Haoxin Ma, Zhengkun Tian, 
    Chenglong Wang, Tao Wang, and Ruibo Fu.
    
    Data can be downloaded here: https://zenodo.org/records/10377492
    Paper can be read here: https://arxiv.org/pdf/2104.03617.pdf

    Args:
        root_dir (str): path to where the data will be downloaded. 
        fs (int): sampling rate of file.
        set_type (str): type of set. usually train, test, or dev.
        remove_zip (bool): remove downloaded zip file. default set to True just to save you some more space.
    '''
    def __init__(self, root_dir:str, fs:int, set_type:str, remove_zip:bool=True) -> None:
        super().__init__()
        self.fs = fs
        self.root_dir = root_dir 
        self.remove_zip = remove_zip
        self.set_type = set_type

        # Check if it is downloaded or not:
        self.data_dir = os.path.join(self.root_dir, 'HAD') 
        is_downloaded = os.path.isdir(self.data_dir) 
        if not is_downloaded:
            download_halftruth_dataset(self.root_dir, self.remove_zip) 

        # Construct additional params from path and metadata:
        self.set_root_path = os.path.join(self.data_dir, f'HAD_{self.set_type}')
        self.text_file_path = os.path.join(self.set_root_path, f'HAD_{self.set_type}_label.txt')
        self.text_file = open(self.text_file_path, 'r').read()
        self.observations = self.text_file.split('\n')

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # Load in observation and get relevant information.
        observation = self.observations[idx] 
        filename, timestamps, _ = observation.split(' ')
        
        # Load file and construct to torch tensor
        audio, _ = librosa.load(os.path.join(self.set_root_path, self.set_type, f'{filename}.wav'), sr=self.fs)
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
        root_dir (str): root directory of data.
        fs (int): sampling rate of file.
        remove_zip (bool): remove downloaded zip file. default set to True just to save you some more space.
    '''
    def __init__(self, root_dir:str, fs:int, remove_zip:bool=True, \
                 ljspeech_dataset_root:str=None, jsut_dataset_root:str=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.fs = fs
        self.remove_zip = remove_zip
        self.ljspeech_datset_root = ljspeech_dataset_root
        self.ljspeech_datset_files = self._check_real_dir(self.ljspeech_datset_root)

        self.jsut_dataset_root = jsut_dataset_root
        self.jsut_dataset_files = self._check_real_dir(self.jsut_dataset_root)

        is_downloaded = os.path.isdir(self.root_dir)
        if not is_downloaded:
            download_wavefake(self.root_dir, self.remove_zip)
    
        # Parse out relevant information:
        self.generated_root_dir = os.path.join(self.root_dir, 'generated_audio')
        self.generated_root_dir_wavs = find_all_wav_files(self.generated_root_dir)

        self.all_files = self.generated_root_dir_wavs + self.ljspeech_datset_files + self.jsut_dataset_files

    def __len__(self): 
        return len(self.all_files)

    def __getitem__(self, idx): 
        # Get filepath:
        filepath = self.all_files[idx]

        # Check if which root dir is in the path.
        # This will serve as our binary labels.
        label = 1 if self.generated_root_dir in filepath else 0

        # Load in audio, and construct sample-wise labels:
        audio, _ = librosa.load(filepath, sr=self.fs)
        audio = torch.from_numpy(audio)
        labels = torch.full(audio.shape, label)

        return audio, labels
    
    def _check_real_dir(self, dataset_root:str) -> list:
        '''
        Check if directory of real files exists. If so, get all wav files in it.

        Args:
            dataset_root (str): path to dataset root.
        Returns:
            list_of_files (list): list of all wav files in root if dataset exists.
        '''
        if dataset_root:
            return find_all_wav_files(dataset_root)
        else:
            return []