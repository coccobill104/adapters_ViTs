import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class VeRALinear(nn.Module):
    def __init__(self, original_layer, A, B, rank = 4):
        ''''
        requires A, B matrices to be given
        '''
        super(VeRALinear, self).__init__()
        self.original_layer = original_layer
        self.A = A
        self.B = B
        self.rank = rank

        self.vera_middle = torch.ones(self.rank, dtype=torch.float32)
        self.vera_out = torch.zeros(original_layer.out_features, dtype=torch.float32)

    def forward(self, x):
        original_output = self.original_layer(x)
        vera_output = self.vera_out * (self.A * self.vera_middle) @ (self.B @ x.T)
        return original_output + vera_output
    


class VeRASelfAttention(nn.Module):
    def __init__(self, original_attn: nn.MultiheadAttention, rank=4, alpha=4.0, qkv = [True, False, False], matrices:dict ={}):
        '''
        q, k, v are bools, they indicate to which blocks to apply the vera
        matrices is a dict with keys A_q, A_k, A_v, B_q, B_k, B_v
        '''
        super().__init__()

        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = original_attn.batch_first # Torchvision ViT usually uses True
        self.scale = self.head_dim ** -0.5
        

        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
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
            self.A_q = nn.Parameter(matrices['A_q'])
            self.B_q = nn.Parameter(matrices['B_q'])

            self.vera_d_q = nn.Parameter(torch.ones(self.rank))
            self.vera_b_q = nn.Parameter(torch.zeros(self.embed_dim))
        if qkv[1]:
            self.A_k = nn.Parameter(matrices['A_k'])
            self.B_k = nn.Parameter(matrices['B_k'])

            self.vera_d_k = nn.Parameter(torch.ones(self.rank))
            self.vera_b_k = nn.Parameter(torch.zeros(self.embed_dim))
        if qkv[2]:
            self.A_v = nn.Parameter(matrices['A_v'])
            self.B_v = nn.Parameter(matrices['B_v'])

            self.vera_d_v = nn.Parameter(torch.ones(self.rank))
            self.vera_b_v = nn.Parameter(torch.zeros(self.embed_dim))

    def forward(self, query, key, value, attn_mask=None, **kwargs):

        is_batched = query.dim() == 3

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        

        if self.which_proj[0]:
            vera_q = (query @ self.A_q.T*self.vera_d_q.T) @ self.B_q.T*self.vera_b_q.T
            q += (vera_q * self.scaling)
        
        if self.which_proj[1]:
            vera_k = (key @ self.A_k.T*self.vera_d_k.T) @ self.B_k.T*self.vera_b_k.T
            k += (vera_k * self.scaling)

        if self.which_proj[2]:
            vera_v = (value @ self.A_v.T*self.vera_d_v.T) @ self.B_v.T*self.vera_b_v.T
            v += (vera_v * self.scaling)
        

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