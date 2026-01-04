# adapters_ViTs
Implementation of LoRAs, VeRAs and more on ViTs

This is a project for the Deep Learning class, taught by Maria Vakalopoulos ana Vincent Lepetit for the MVA master. Our goal is to implement an assortment of efficient adapters on a pre-trained ViT and study the effectiveness of various trainig regiems depending on parameters. 

Adapters: 
- LoRAs https://arxiv.org/abs/2106.09685
- VeRas https://arxiv.org/abs/2310.11454
- IA3  https://arxiv.org/abs/2205.05638

Benchmarks: inspired from
- VTAB-1k https://arxiv.org/abs/1910.04867; https://github.com/BenediktAlkin/vtab1k-pytorch




Plan: 

Tasks: preidction performance on some (or all) datasets

For each task apply: linear, LoRA (various r, mlps and attention), VeRA (various r, mlps and attention), IA3

On 1 or 2 task try to do full fine tuning (set all params as requires_grad=True)

Try 1 seed on each task, if possible try 3 on some of them

Keep track of: 1. number and percentage of trainable parameters, 2. epoch time, 3.VRAM peak




Requirements:

torch
torchvision
yaml
tqdm


# File structure:
mini_vtab/
  configs/
    cifar100.yaml
    eurosat.yaml
    dtd.yaml
  data/
    datasets.py
  models/
    build.py
  adapters/
    IA3.py
    LoRAs.py
    PEFTclass.py
    VeRAs.py
  utils/
    metrics.py
    seed.py
    logging.py
  train.py
  run_suite.py
  requirements.txt



# File descritpion

- **run_suite.py**: runs the experiment
- **train.py**: handles training

- **sanity_check.py**: generic sanity check for the pipeline
- **control_panel.ipynb**: now empty
- **leftovers.py**: old sanity checkers, might be useful later on?


**ADAPTERS**
- **PEFT.py**: contains the PEFTViT class, the main object of the implementation. 
PEFTViT takes a trained ViT, substitues the head with a head of the correct size and applies the required adapter. 
Important features:
1. set_trainable_parameters: method that freezes vit parameters, allows grad for head and adapter
2. state_dict and load_state_dict are overridden, they only act on parameters requiring gradient. 
- **LoRA.py, VeRA.py, IA3.py**: contain implementation and wrappers of the adapters on linear and attention layers.
