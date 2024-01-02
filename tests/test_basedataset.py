import torch 
from data.torch_datasets import BaseDataset 

VECTOR_LEN_SAMPLES = 160001 
VECTOR = torch.rand(VECTOR_LEN_SAMPLES)

DURATION_SEC = 2
FS = 16000
HOP_SIZE = 128
WIN_SIZE = 128

DATASET = BaseDataset(DURATION_SEC, FS, HOP_SIZE, WIN_SIZE) 

def test_pad():
    '''Test pad'''
    padded_vector = DATASET._pad_vector(VECTOR)
    assert len(padded_vector) == 160128, f'Padding in samples is not correct. Expected 160128 but got {len(padded_vector)}'

def test_reflection():
    '''Test reflection'''
    padded_vector = DATASET._pad_vector(VECTOR)
    assert padded_vector[0] == padded_vector[1], 'Reflection padding does not work on left padding.'
    assert padded_vector[-1] == padded_vector[-2], 'Reflection padding does not work on right padding.'

def test_framing():
    '''Test framing.'''
    padded_vector = DATASET._pad_vector(VECTOR)
    framed_vector = DATASET._frame_vector(padded_vector)
    assert len(framed_vector) == 1251, f'Number of frames is not expected. Expected 1251 but got {len(framed_vector)}'