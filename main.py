import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_vit import ViT

#pretrained vit
#model = ViT('B_16_imagenet1k', pretrained=True)



class LoRAlinear(nn.Module):
    def __init__(self, original_layer, rank = 4, alpha = 1.0):
        super(LoRAlinear, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.randn(original_layer.out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (self.lora_A @ self.lora_B) @ x.T
        lora_output = lora_output.T * (self.alpha / self.rank)
        return original_output + lora_output
    



class VeRAlinear(nn.Module):
    def __init__(self, original_layer, A, B, rank = 4):
        super(VeRAlinear, self).__init__()
        self.original_layer = original_layer
        self.A = A
        self.B = B
        self.rank = rank

        self.middle = torch.ones(self.rank, dtype=torch.float32)
        self.out = torch.zeros(original_layer.out_features, dtype=torch.float32)

    def forward(self, x):
        original_output = self.original_layer(x)
        vera_output = self.out * (self.A * self.middle) @ (self.B @ x.T)
        return original_output + vera_output