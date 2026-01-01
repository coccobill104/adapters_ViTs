import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision.models import ViT_B_16_Weights

from PEFTclass import *

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





#   Method for checking basic properties of the PEFT class, already used but possibly useful in the future
# BY GEMINI
def check_PEFTViT():
    print("--- 1. SETUP: Loading Base Model ---")
    base_model = torchvision.models.vit_b_16(weights=weights)

    
    print("\n--- 2. WRAP: Creating PEFTViT ---")
    # Wrap the model
    #model = PEFTViT(base_model, nb_classes = 10, method='vera', r=4)
    model = PEFTViT(base_model, nb_classes = 10, method='ia3')
    
    print("\n--- 3. FREEZE: Setting Gradients ---")
    # Simulate the freezing logic: Freeze everything EXCEPT head and LoRA
    model.set_trainable_parameters()
            
    model.print_trainable_parameters()
    # Expected: Small % (only head + lora params)

    print("\n--- 4. MOCK TRAINING (Modifying Weights) ---")
    # We manually modify a weight to prove saving works
    with torch.no_grad():
        # Modify one of the LoRA parameters
        #original_val = model.model.encoder.layers[0].mlp[0].lora_A[0, 0].item()
        #model.model.encoder.layers[0].mlp[0].lora_A[0, 0] += 1.0
        #new_val = model.model.encoder.layers[0].mlp[0].lora_A[0, 0].item()

        #original_val = model.model.encoder.layers[0].mlp[0].vera_middle[0].item()
        #model.model.encoder.layers[0].mlp[0].vera_middle[0] += 1.0
        #new_val = model.model.encoder.layers[0].mlp[0].vera_middle[0].item()

        original_val = model.model.encoder.layers[0].mlp[3].ia3_vector[0].item()
        model.model.encoder.layers[0].mlp[3].ia3_vector[0] += 1.0
        new_val = model.model.encoder.layers[0].mlp[3].ia3_vector[0].item()
    print(f"   Modified lora_A[0,0]: {original_val:.4f} -> {new_val:.4f}")

    print("\n--- 5. SAVE: Testing state_dict override ---")
    #save_path = "temp_lora_checkpoint.pt"
    #save_path = "temp_vera_checkpoint.pt"
    save_path = "temp_ia3_checkpoint.pt"

    # This calls our custom state_dict() method!
    torch.save(model.state_dict(), save_path)
    
    file_size = os.path.getsize(save_path) / 1024
    print(f"   Saved checkpoint size: {file_size:.2f} KB")
    # If this were the full model, it would be ~45MB. 
    # Since it is <100KB, we know it saved ONLY the adapter.

    print("\n--- 6. LOAD: Testing load_state_dict override ---")
    # Create a fresh model (original weights) to prove we are loading the changes
    fresh_base = torchvision.models.vit_b_16(weights=weights)
    fresh_base.head = nn.Linear(512, 100)
    
    # Wrap it
    #new_model = PEFTViT(fresh_base, nb_classes=10, method='vera', r=4)
    new_model = PEFTViT(fresh_base, nb_classes=10, method='ia3')
    
    # Verify it has the OLD value before loading
    #print(f"   Value before load: {new_model.model.encoder.layers[0].mlp[0].lora_A[0, 0].item():.4f}")
    #print(f"   Value before load: {new_model.model.encoder.layers[0].mlp[0].vera_middle[ 0].item():.4f}")
    print(f"   Value before load: {new_model.model.encoder.layers[0].mlp[3].ia3_vector[0].item():.4f}")
    
    
    # Load the saved adapter
    # This calls our custom load_state_dict() method!
    saved_weights = torch.load(save_path)
    new_model.load_state_dict(saved_weights)
    
    #print(f"   Value after load:  {new_model.model.encoder.layers[0].mlp[0].lora_A[0, 0].item():.4f}")
    #print(f"   Value after load:  {new_model.model.encoder.layers[0].mlp[0].vera_middle[0].item():.4f}")
    print(f"   Value after load:  {new_model.model.encoder.layers[0].mlp[3].ia3_vector[0].item():.4f}")
    
    #if abs(new_model.model.encoder.layers[0].mlp[0].lora_A[0, 0].item() - new_val) < 1e-5:
    if abs(new_model.model.encoder.layers[0].mlp[3].ia3_vector[0].item() - new_val) < 1e-5:
        print("\n✅ SUCCESS: Weights restored correctly.")
    else:
        print("\n❌ FAILURE: Weights do not match.")
        
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)




#function to check basic properties of the adapters. Already used but possible useful
#BY GEMINI
def run_tests():
    print("--- Starting LoRA Implementation Tests ---")
    
    # Load Model (dummy weights are fine, but we use ImageNet for realism)
    print("1. Loading ViT-B/16...")
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Isolate one attention layer
    original_layer = model.encoder.layers[0].self_attention
    
    # Create dummy input (Batch=1, Seq=197, Dim=768)
    # 197 = 1 class token + (14x14) patches
    dummy_input = torch.randn(1, 197, 768)
    
    # ---------------------------------------------------------
    # Test A: Does the new layer produce the exact same output?
    # ---------------------------------------------------------
    print("\nTest A: Output Consistency (Original vs LoRA with B=0)")
    
    # Init LoRA layer
    lora_layer = LoRASelfAttention(original_layer)

    matrices = { 'A_q': torch.randn(( 4, 768)), 'B_q': torch.randn((768, 4))}
    vera_layer = VeRASelfAttention(original_layer, matrices=matrices)

    print('lora qkv', lora_layer.which_proj)
    print('vera qkv', vera_layer.which_proj)
    
    # Set to eval mode to disable dropout in both
    original_layer.eval()
    lora_layer.eval()
    vera_layer.eval()
    
    with torch.no_grad():
        # Original Forward (Pass x, x, x manually or rely on its forward)
        # torchvision ViT passes x, x, x internaly
        out_orig, _ = original_layer(dummy_input, dummy_input, dummy_input, need_weights=False)
        
        # LoRA Forward
        out_lora, _ = lora_layer(dummy_input, dummy_input, dummy_input)
        out_vera, _ = vera_layer(dummy_input, dummy_input, dummy_input)
        
    # Check difference
    diff = torch.abs(out_orig - out_lora).max().item()
    diff_vera = torch.abs(out_orig - out_vera).max().item()
    print(f"   Max absolute difference lora: {diff:.8f}")
    print(f"   Max absolute difference vera: {diff_vera:.8f}")
    
    if diff < 1e-5:
        print("   ✅ LORA SUCCESS: Outputs match closely.")
    else:
        print("   ❌ LORA FAILURE: Outputs diverge too much.")

    if diff_vera < 1e-5:
        print("   ✅ VERA SUCCESS: Outputs match closely.")
    else:
        print("   ❌ VERA FAILURE: Outputs diverge too much.")


    # ---------------------------------------------------------
    # Test B: Does LoRA actually modify the output when trained?
    # ---------------------------------------------------------
    print("\nTest B: LoRA Activation Check")
    
    # Manually set LoRA B matrix to something non-zero to simulate training
    with torch.no_grad():
        lora_layer.lora_B_q.fill_(1.0)
    
    with torch.no_grad():
        out_lora_active, _ = lora_layer(dummy_input, dummy_input, dummy_input)
        
    diff_active = torch.abs(out_orig - out_lora_active).max().item()
    print(f"   Diff after activating LoRA: {diff_active:.4f}")
    
    if diff_active > 1e-3:
        print("   ✅ SUCCESS: LoRA is modifying the output.")
    else:
        print("   ❌ FAILURE: LoRA parameters are not affecting the output.")

    print("\nTest B: vera Activation Check")
    


    # Manually set VeRA b vector to something non-zero to simulate training
    with torch.no_grad():
        vera_layer.vera_b_q.fill_(1.0)
    
    with torch.no_grad():
        out_vera_active, _ = vera_layer(dummy_input, dummy_input, dummy_input)
        
    diff_active_vera = torch.abs(out_orig - out_vera_active).max().item()
    print(f"   Diff after activating LoRA: {diff_active_vera:.4f}")
    
    if diff_active_vera > 1e-3:
        print("   ✅ SUCCESS: VeRA is modifying the output.")
    else:
        print("   ❌ FAILURE: VeRA parameters are not affecting the output.")

    # ---------------------------------------------------------
    # Test C: Full Model Integration (Swapping the layer)
    # ---------------------------------------------------------
    print("\nTest C: Full Model Integration")
    
    # Swap the layer in the model
    model.encoder.layers[0].self_attention = lora_layer
    
    # Run full model forward pass
    try:
        logits = model(torch.randn(1, 3, 224, 224)) # Standard ImageNet input
        print(f"   Output shape: {logits.shape}")
        print("   ✅ SUCCESS: Full model forward pass complete.")
    except Exception as e:
        print(f"   ❌ FAILURE: Model crashed with error: {e}")


    # Swap the layer in the model
    model.encoder.layers[0].self_attention = vera_layer
    
    # Run full model forward pass
    try:
        logits = model(torch.randn(1, 3, 224, 224)) # Standard ImageNet input
        print(f"   Output shape: {logits.shape}")
        print("   ✅ SUCCESS: Full model forward pass complete.")
    except Exception as e:
        print(f"   ❌ FAILURE: Model crashed with error: {e}")

    # ---------------------------------------------------------
    # Test D: Parameter Freeze Check
    # ---------------------------------------------------------
    print("\nTest D: Parameter Counting")
    
    model.encoder.layers[0].self_attention = lora_layer

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only LoRA params
    lora_params = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params += param.numel()
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total trainable parameters: {trainable_params}")
    
    if trainable_params == lora_params and lora_params > 0:
        print("   ✅ SUCCESS: Only LoRA parameters are trainable.")
    else:
        print(f"   ❌ FAILURE: Trainable params ({trainable_params}) != LoRA params ({lora_params})")




    model.encoder.layers[0].self_attention = vera_layer
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only LoRA params
    vera_params = 0
    for name, param in model.named_parameters():
        if "vera_" in name:
            param.requires_grad = True
            vera_params += param.numel()
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total trainable parameters: {trainable_params}")
    
    if trainable_params == vera_params and vera_params > 0:
        print("   ✅ SUCCESS: Only VeRA parameters are trainable.")
    else:
        print(f"   ❌ FAILURE: Trainable params ({trainable_params}) != VeRA params ({vera_params})")