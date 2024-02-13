from audio_sleuth.data.datasets import WaveFake, HalfTruthDataset

ROOT_DIR = 'audio_data/wavefake'
FS = 16000

def test_wavefake():
    wavefake = WaveFake(ROOT_DIR, FS)
    audio, labels = wavefake[0]
    assert audio.shape == labels.shape
