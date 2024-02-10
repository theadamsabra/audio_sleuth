from audio_sleuth.data.datasets import HalfTruthDataset
import warnings
warnings.simplefilter('ignore')

ROOT_DIR = 'audio_data/half_truth'
FS = 44100

def test_halftruth():
    train = HalfTruthDataset(ROOT_DIR, FS, 'train')
    audio, labels = train[0]
    assert audio.shape == labels.shape
    assert len(audio) == 408543