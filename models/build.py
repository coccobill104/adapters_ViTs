import torchvision
from torchvision.models import ViT_B_16_Weights

# importa il tuo wrapper
from adapters.PEFTclass import PEFTViT  

def build_model(num_classes: int, method: str, adapter_cfg: dict): 


    vit = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    model = PEFTViT(vit, method = method, nb_classes=num_classes, **adapter_cfg)

    method = method.lower()

#    if method == "linear":
#        model.freeze_backbone()  # solo head
#    else:
#        model.apply_adapter(method, **adapter_cfg)
#        model.freeze_backbone()
#        model.unfreeze_adapter_and_head()

    return model
