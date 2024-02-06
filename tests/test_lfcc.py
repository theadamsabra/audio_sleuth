import torch
from audio_sleuth.augmentations import LFCC 

VECTOR = torch.randn(160000)
LABELS = torch.randn(160000)

FS = 16000 
N_FILTERS = 128
N_LFCC = 40

def lfcc_helper(n_fft, hop_size, win_size):
    lfcc = LFCC(FS, n_fft, hop_size, win_size, N_FILTERS, N_LFCC)
    lfcc_output, labels = lfcc(VECTOR, LABELS)
    assert lfcc_output.shape[-1] == labels.shape[0], f'Mismatch of frame lengths, LFCC output is {lfcc_output.shape}, whereas labels is {labels.shape}.'

def test_lfcc():
    lfcc_helper(1024, 512, 256)
    lfcc_helper(400, 100, 300)
    lfcc_helper(512, 512, 512)
    