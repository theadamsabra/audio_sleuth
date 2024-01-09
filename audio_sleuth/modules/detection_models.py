import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class Wav2Vec2FramewiseDetection(nn.Module):
    def __init__(self, wav2vec2_huggingface_id:str, num_classes:int) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            wav2vec2_huggingface_id,
            num_labels=num_classes
        )
        self.feature_extractor = Wav2Vec2Model(self.config)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.final_dropout)
        self.out_proj = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, waveform:Tensor) -> Tensor:
        x = self.feature_extractor(waveform)[0]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x 