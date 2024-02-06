import sys
sys.path.append('..')

import torch
from audio_sleuth.augmentations import LabelAlignment

fs = 44100
VECTOR = torch.randn(fs*10) # 10 seconds of 44.1 kHz

FAKE_SEGMENT_START = 3
FAKE_SEGMENT_END = 5

LABELS = torch.full((fs*10,1), 0)
LABELS[FAKE_SEGMENT_START*fs:FAKE_SEGMENT_END*fs] = 1


def label_alignment_helper(win_size, hop_size, vector, desired_shape):
    label_alignment = LabelAlignment(win_size, hop_size)
    framed_vector = label_alignment(vector)
    assert framed_vector.shape == torch.Size([desired_shape]), f'Shape mismatch, output shape is {framed_vector.shape}'


def test_label_alignment():
    label_alignment_helper(512, 512, VECTOR, 862)
    label_alignment_helper(512, 256, VECTOR, 1723)
    label_alignment_helper(256, 256, VECTOR, 1723)
    label_alignment_helper(1024, 512, VECTOR, 862)