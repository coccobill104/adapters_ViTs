import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class IA3Linear(nn.Module):
    '''
    By design, this only gets applied to the second linear layer in the encoder blocks
    could try something more
    '''
    def __init__(self, original_linear_layer):
        super().__init__()
        self.original_linear_layer = original_linear_layer
        
        n_out = original_linear_layer.out_features
        self.ia3_vector = nn.Parameter(torch.ones(n_out)) 

    def forward(self, x):

        #might be wrong maybe it's self.ia3_vector+X
        new_input =  x*self.ia3_vector

        output = self.original_linear_layer(new_input)
        return output



class IA3SelfAttention(nn.Module):
    def __init__(self, original_attn: nn.MultiheadAttention, alpha=4.0, qkv = [False, True, True]):
        '''
        qkv is a list of bools, they indicate to which blocks to apply the lora
        '''
        super().__init__()

        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = original_attn.batch_first # Torchvision ViT usually uses True
        self.scale = self.head_dim ** -0.5
        

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
            self.ia3_q = nn.Parameter( torch.randn(self.embed_dim))
        if qkv[1]:
            self.ia3_k = nn.Parameter( torch.randn(self.embed_dim))
        if qkv[2]:
            self.ia3_v = nn.Parameter( torch.randn(self.embed_dim))


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
            q = self.ia3_q*q
        
        if self.which_proj[1]:
            k = self.ia3_k*k

        if self.which_proj[2]:
            v = self.ia3_v*v

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
