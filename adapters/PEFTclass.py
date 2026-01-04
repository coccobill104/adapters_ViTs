import torch.nn as nn
import copy

from adapters.LoRAs import *
from adapters.VeRAs import *
from adapters.IA3 import *



class PEFTViT(nn.Module):
    def __init__(self, base_model, nb_classes=1000, method='lora', **kwargs):
        '''
        Applies the adapter method to the model and replaces the head with the correct-dimensional layer
        '''
        super().__init__()
        self.method = method        

        if method == 'linear':
            self.model = base_model
        if method == 'lora':
            self.model = apply_LoRA(base_model, **kwargs)
        elif method == 'vera':
            self.model = apply_VeRA(base_model, **kwargs)
        elif method == 'ia3':
            self.model = apply_IA3(base_model, **kwargs)
        else:
            self.model = copy.deepcopy(base_model)

        self.model.heads = nn.Linear(768, nb_classes, bias = True)

    def forward(self, x):
        return self.model(x)


    def set_trainable_parameters(self):
        '''
        Sets parameters of the adapters and of the head to requires_grad=True, all other parameters to False
        '''
        for name, param in self.model.named_parameters():
            if "head" in name or self.method in name:
                param.requires_grad = True
            else:
                param.requires_grad = False



    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        percent = 100 * trainable_params / all_param if all_param > 0 else 0
        print(f"[Info] Trainable: {trainable_params:,} | Total: {all_param:,} | %: {percent:.2f}%")



    def state_dict(self, destination=None, prefix='', keep_vars=False, print_percentage=False):
        """
        override of the classical state_dict method, only shows trainable parameters
        """
        full_state_dict = self.model.state_dict(destination, prefix, keep_vars)
        
        filtered_dict = {
            k: v for k, v in full_state_dict.items() 
            if f"{k}" in [n for n, p in self.model.named_parameters() if p.requires_grad]
        }
        
        if print_percentage:
            print(f"[Save] Filtering state_dict... Keeping {len(filtered_dict)}/{len(full_state_dict)} keys.")
        return filtered_dict



    def load_state_dict(self, state_dict, strict=False):
        """
        OVERRIDE: Forces strict=False to ignore missing base model weights.
        """
        # We force strict=False so it doesn't complain about missing base weights
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if len(unexpected) > 0:
            print(f" Unexpected keys found: {unexpected}")
        else:
            print("Adapter weights merged into base model.")



