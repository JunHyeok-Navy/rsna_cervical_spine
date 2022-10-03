import timm
import torch
import config

class EffnetModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.effnet = model

        self.nn_fracture = torch.nn.Sequential(
            torch.nn.LSTM(1000, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.LSTM(1000, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.effnet(x)
        x = x.unsqueeze(0)
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)