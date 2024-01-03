import torch
from audio_sleuth.augmentations import ResampleBlock

VECTOR = torch.randn(441000)
LABELS = torch.randn(441000)
INPUT_SR = 44100
NEW_SR = 32000

def test_resample_to_from_sr():
    RESAMPLE = ResampleBlock(INPUT_SR, NEW_SR, return_original_sr=True)
    vector, labels = RESAMPLE(VECTOR, LABELS)
    assert vector.shape == labels.shape, 'Arrays are not the same shape after resampling and returning to original sampling rate.'


def test_resample_to_sr():
    RESAMPLE = ResampleBlock(INPUT_SR, NEW_SR, return_original_sr=False)
    vector, labels = RESAMPLE(VECTOR, LABELS)
    assert vector.shape == labels.shape, 'Arrays are not the same shape after resampling and returning to original sampling rate.'