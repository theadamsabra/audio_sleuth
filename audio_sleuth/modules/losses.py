import torch
import torch.nn as nn
import torch.nn.functional as F

class OCSoftmax(nn.Module):
    '''
    One Class Softmax implemented from:
        One-class Learning Towards Synthetic Voice Spoofing Detection
            by
        You Zhang, Fei Jiang, and Zhiyao Duan.
    
    Core code is from [here](https://github.com/yzyouzhang/AIR-ASVspoof/blob/master/loss.py) and is repurposed
    from  frame-level prediction.
    '''
    def __init__(self, feat_dim:int=2, r_real:float=0.9, r_fake:float=0.5, alpha:float=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)