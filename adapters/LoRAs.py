import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank = 4, alpha = 1.0):
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        self.lora_B = nn.Parameter(torch.randn(original_layer.out_features, rank) * 0.01)
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)

    def forward(self, x):
        original_output = self.original_layer(x)
        #lora_output = (self.lora_A @ self.lora_B) @ x.T
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * (self.alpha / self.rank)
        return original_output + lora_output
    


class LoRASelfAttention(nn.Module):
    def __init__(self, original_attn: nn.MultiheadAttention, rank=4, alpha=4.0, qkv = [True, False, False]):
        '''
        qkv is a list of bools, they indicate to which blocks to apply the lora
        '''
        super().__init__()

        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = original_attn.batch_first # Torchvision ViT usually uses True
        self.scale = self.head_dim ** -0.5
        

        self.rank = rank
        self.lora_alpha = alpha
        self.scaling = self.lora_alpha / self.rank
        self.which_proj = qkv


        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = original_attn.out_proj 


        with torch.no_grad():
            fused_w = original_attn.in_proj_weight
            fused_b = original_attn.in_proj_bias
            
            self.q_proj.weight.copy_(fused_w[:self.embed_dim, :])
            self.k_proj.weight.copy_(fused_w[self.embed_dim:2*self.embed_dim, :])
            self.v_proj.weight.copy_(fused_w[2*self.embed_dim:, :])
            
            if fused_b is not None:
                self.q_proj.bias.copy_(fused_b[:self.embed_dim])
                self.k_proj.bias.copy_(fused_b[self.embed_dim:2*self.embed_dim])
                self.v_proj.bias.copy_(fused_b[2*self.embed_dim:])


        if qkv[0]:
            self.lora_A_q = nn.Parameter(torch.randn(rank, self.embed_dim)*0.01)
            self.lora_B_q = nn.Parameter(torch.zeros(self.embed_dim, rank))

        if qkv[1]:
            self.lora_A_k = nn.Parameter(torch.randn(rank, self.embed_dim)*0.01)
            self.lora_B_k = nn.Parameter(torch.zeros(self.embed_dim, rank))

        if qkv[2]:
            self.lora_A_v = nn.Parameter(torch.randn(rank, self.embed_dim)*0.01)
            self.lora_B_v = nn.Parameter(torch.zeros(self.embed_dim, rank))




    def forward(self, query, key, value, attn_mask=None, **kwargs):
        """
        In ViT Encoder: query, key, and value are usually the same tensor 'x'.
        Shape of input: (Batch, Seq_Len, Embed_Dim) if batch_first=True
        """
        is_batched = query.dim() == 3

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        

        if self.which_proj[0]:
            lora_q = (query @ self.lora_A_q.T) @ self.lora_B_q.T
            q += (lora_q * self.scaling)
        
        if self.which_proj[1]:
            lora_k = (key @ self.lora_A_k.T) @ self.lora_B_k.T
            k += (lora_k * self.scaling)

        if self.which_proj[2]:
            lora_v = (value @ self.lora_A_v.T) @ self.lora_B_v.T
            v += (lora_v * self.scaling)
        

        B, L, E = q.shape
        
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, E)
        

        output = self.out_proj(attn_output)
        
        return output, None



def apply_LoRA(model, r=4, mlps:bool = True, attention: bool =False, qkv: dict =[False, False, False]):
    '''
    r = rank of the Lora
    mlps = True if lora to be applied on mlp layers
    attention = True if lora to be applied on self attention layers
    qkv = where to apply lora, only relevant if attention=True
    '''
    new_model = copy.deepcopy(model)
    layers = new_model.encoder.layers
    for layer in layers:
        if mlps:
            layer.mlp[0] = LoRALinear(layer.mlp[0], rank=r)

            layer.mlp[3] = LoRALinear(layer.mlp[3], rank=r)

        if attention:
            layer.self_attention = LoRASelfAttention(layer.self_attention, rank=r, qkv = qkv)
    return new_model


