# About Augmentations

All augmentations are `torch.nn.Module` classes which serve as light wrappers to `torchaudio.transforms` with a label aligner from `audio_sleuth.augmentations.LabelAlignment` class. The overarching goal of these augmentations is to ensure that in the time domain we have sample-wise labels while also ensuring that these labels are framed and averaged irrespective of the transformations done on the audio.

## Label Alignment in a Nutshell

In short, the `LabelAlignment` class handles the framing and averaging of the sample-wise labels generated from the datasets in `audio_sleuth.data.datasets`. The only parameters needed for `LabelAlignment` is `win_size` and `hop_size`. For example:

```python
import torch
from audio_sleuth.augmentations import LabelAlignment

audio_example = torch.randn(160000) # Assume 10 sec. @ 16 kHz

hop_size = 512
win_size = 256
label_aligner = LabelAlignment(hop_size=hop_size, win_size=win_size)

print(label_aligner(audio_example).shape)
# >> torch.Size([626])
```

`LabelAlignment` also handles padding. After reading the code of [`torch.stft`](https://github.com/pytorch/pytorch/blob/64aaa8f50847f3b82beac22f291acb9d182221ff/torch/functional.py#L659), it was important this class mimics the padding decision as we ensure there is no delay in label framing. 