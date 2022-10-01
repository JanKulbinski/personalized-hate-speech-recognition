import torch.nn.functional as F
from torch import nn


class MLP_Model(nn.Module):
    def __init__(
        self, name: str = "mlp", features: int = 106, hidden: int = 64, classes: int = 2
    ):
        super().__init__()
        self.name = name
        self.features = features
        self.hidden = hidden
        self.classes = classes

        self.linear1 = nn.Linear(self.features, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.classes)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out
