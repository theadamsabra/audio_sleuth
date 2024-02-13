from audio_sleuth.data.datasets import WaveFake, HalfTruthDataset

ROOT_DIR = 'audio_data/wavefake'
FS = 16000

def test_wavefake_with_only_generated_data():
    wavefake = WaveFake(ROOT_DIR, FS)
    audio, labels = wavefake[0]
    assert audio.shape == labels.shape
    assert labels[0].item() == 1