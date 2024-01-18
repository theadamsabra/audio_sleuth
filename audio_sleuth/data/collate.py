import torch
from torch.utils.data import DataLoader
from torch.nn import Module, Tensor

def pad_and_transform_collate(data_loader:DataLoader, transform:Module=None) -> tuple[Tensor, Tensor, list, list]:
    # Setup all necessary output params
    processed_audio = []
    final_labels = []

    # Loop through batch and run transforms:
    for audio, labels in data_loader:
        # Run transform if defined:
        if transform:
           audio, labels = transform(audio, labels)
        processed_audio.append(audio) 
        final_labels.append(labels)
    
    processed_audio = torch.nn.utils.rnn.pad_sequence(processed_audio, batch_first=True).unsqueeze(1).transpose(2, 3)
    final_labels = torch.nn.utils.rnn.pad_sequence(final_labels, batch_first=True)
    return processed_audio, final_labels.long()