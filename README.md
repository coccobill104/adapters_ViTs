# adapters_ViTs
Implementation of LoRAs, VeRAs and more on ViTs

This is a project for the Deep Learning class, taught by Maria Vakalopoulos ana Vincent Lepetit for the MVA master. Our goal is to implement an assortment of efficient adapters on a pre-trained ViT and study the effectiveness of various trainig regiems depending on parameters. 

Adapters: 
- LoRAs https://arxiv.org/abs/2106.09685
- VeRas https://arxiv.org/abs/2310.11454
- IA$ ^3$  https://arxiv.org/abs/2205.05638

Benchmarks: 
- VTAB-1k https://arxiv.org/abs/1910.04867; https://github.com/BenediktAlkin/vtab1k-pytorch


Requirements:

torch
torchvision



File structure:

- **LoRA.py, VeRA.py, IA3.py**: contain implementation of the adapters on linear and attention layers. Wrappers are currently in the **control_panel.ipynb** file.

- **control_panel.ipynb**: contains wrappers for the adapters