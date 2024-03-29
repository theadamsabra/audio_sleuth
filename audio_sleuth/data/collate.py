import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import Module

def pad_and_transform_collate(data:tuple, transform:Module=None) -> tuple[Tensor, Tensor]:
    '''
    Implementation from 
        https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
    
    Args:
        data_loader (Dataloader): dataloader for torch Dataset.
        transform (Module): any designed augmentations from audio_sleuth.augmentations.py
    
    Return:
        processed_audio (Tensor): tensor of size (batch_size, feature_size, num_frames). num_frames represents
        the number of frames from the longest audio file in batch.

        final_labels (Tensor): final encoded labels 
    '''

    # Setup all necessary output params
    processed_audio = []
    final_labels = []

    # Loop through batch and run transforms:
    for audio, labels in data:
        # Run transform if defined:
        if transform:
           audio, labels = transform(audio, labels)

        processed_audio.append(audio.T) 
        final_labels.append(labels)

    # Dynamic padding to fit to the longest segment in batch. 
    if transform:
        # Feature space padding
        processed_audio = torch.nn.utils.rnn.pad_sequence(processed_audio, batch_first=True).unsqueeze(1).transpose(2, 3)
    else:
        # Time domain padding
        processed_audio = torch.nn.utils.rnn.pad_sequence(processed_audio, batch_first=True).unsqueeze(1)

    final_labels = torch.nn.utils.rnn.pad_sequence(final_labels, batch_first=True)
    return processed_audio, final_labels.long()