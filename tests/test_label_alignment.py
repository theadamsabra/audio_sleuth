import torch 
from audio_sleuth.augmentations import LabelAlignment

VECTOR_LEN_SAMPLES = 160001 
VECTOR = torch.rand(VECTOR_LEN_SAMPLES)

FS = 16000
HOP_SIZE = 128
WIN_SIZE = 128

label_alignment = LabelAlignment(HOP_SIZE, WIN_SIZE)

def test_pad():
    '''Test pad'''
    padded_vector = label_alignment._pad_vector(VECTOR)
    assert len(padded_vector) == 160128, f'Padding in samples is not correct. Expected 160128 but got {len(padded_vector)}'

def test_reflection():
    '''Test reflection'''
    padded_vector = label_alignment._pad_vector(VECTOR)
    assert padded_vector[0] == padded_vector[1], 'Reflection padding does not work on left padding.'
    assert padded_vector[-1] == padded_vector[-2], 'Reflection padding does not work on right padding.'

def test_framing():
    '''Test framing.'''
    padded_vector = label_alignment._pad_vector(VECTOR)
    framed_vector = label_alignment._frame_vector(padded_vector)
    assert len(framed_vector) == 1251, f'Number of frames is not expected. Expected 1251 but got {len(framed_vector)}'