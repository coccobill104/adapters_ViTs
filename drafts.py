import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision.models import ViT_B_16_Weights

#from torchvision.datasets import Flowers102
#from torchvision import transforms
#from torchvision.datasets import Food101


# Example: Injecting IA3 into the Attention Output of the first block
# The path to the layer in torchvision's ViT is usually:
# model.encoder.layers[i].self_attention.out_proj

#target_layer = model.encoder.layers[0].self_attention.out_proj
#model.encoder.layers[0].self_attention.out_proj = IA3Layer(target_layer)

# Now, only 'ia3_vector' is trainable in this layer!





# Load the model with the best available pre-trained weights
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = torchvision.models.vit_b_16(weights=weights)

# Freeze the base model immediately
for param in model.parameters():
    param.requires_grad = False



# Define the standard ImageNet transforms (required for the pre-trained ViT)
#transform = transforms.Compose([
#    transforms.Resize((224, 224)), # ViT requires 224x224
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])

# detailed split: 'train', 'val', or 'test'
#train_dataset = Flowers102(root='./data', split='train', download=True, transform=transform)
#val_dataset = Flowers102(root='./data', split='val', download=True, transform=transform)



# Food101 is larger (5GB+), so make sure you have disk space on Colab
#train_dataset = Food101(root='./data', split='train', download=True, transform=transform)
#test_dataset = Food101(root='./data', split='test', download=True, transform=transform)