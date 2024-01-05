import torch
import torch.nn as nn
from torch import Tensor
from transformers import Wav2Vec2ForPreTraining 

class Wav2Vec2FramewiseDetection(nn.Module):
    def __init__(self, wav2vec2_huggingface_id:str, wav2vec2_dimensionality:int, \
                num_classes:int, activation:nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_extractor = Wav2Vec2ForPreTraining.from_pretrained(wav2vec2_huggingface_id)
        assert num_classes > 1, "How are you predicting less than two classes? num_classes greater than 1."
        self.linear = nn.Linear(in_features=wav2vec2_dimensionality, out_features=num_classes)   
        self.activation = activation 

    def forward(self, waveform:Tensor) -> Tensor:
        features = self.feature_extractor(waveform)
        output = self.activation(self.linear(features.projected_states))
        return output 