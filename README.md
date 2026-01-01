# adapters_ViTs
Implementation of LoRAs, VeRAs and more on ViTs

This is a project for the Deep Learning class, taught by Maria Vakalopoulos ana Vincent Lepetit for the MVA master. Our goal is to implement an assortment of efficient adapters on a pre-trained ViT and study the effectiveness of various trainig regiems depending on parameters. 

Adapters: 
- LoRAs https://arxiv.org/abs/2106.09685
- VeRas https://arxiv.org/abs/2310.11454
- IA3  https://arxiv.org/abs/2205.05638

Benchmarks: 
- VTAB-1k https://arxiv.org/abs/1910.04867; https://github.com/BenediktAlkin/vtab1k-pytorch


Requirements:

torch
torchvision



File structure:


- **PEFT.py**: contains the PEFTViT class, the main object of the implementation. 
PEFTViT takes a trained ViT, substitues the head with a head of the correct size and applies the required adapter. 
Important features:
1. set_trainable_parameters: method that freezes vit parameters, allows grad for head and adapter
2. state_dict and load_state_dict are overridden, they only act on parameters requiring gradient. 

- **LoRA.py, VeRA.py, IA3.py**: contain implementation and wrappers of the adapters on linear and attention layers.

- **control_panel.ipynb**: now empty