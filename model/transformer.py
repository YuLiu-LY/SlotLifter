import torch
import torch.nn as nn
import torch.nn.functional as F


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1., nonlinearity='relu'):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_q, d_kv, d_proj, d_out, num_heads, dropout=0., gain=1., bias=True):
        super().__init__()
        
        assert d_proj % num_heads == 0, "d_proj must be divisible by num_heads"
        self.num_heads = num_heads
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_q, d_proj, bias=bias)
        self.proj_k = linear(d_kv, d_proj, bias=bias)
        self.proj_v = linear(d_kv, d_proj, bias=bias)
        self.proj_o = linear(d_proj, d_out, bias=bias, gain=gain)
    
    
    def forward(self, q, k, v, attn_mask=None):
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn_d = self.attn_dropout(attn)
        
        output = torch.matmul(attn_d, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output
    

class TransformerDecoderBlock(nn.Module):
    
    def __init__(self, d_q, d_kv, num_heads=4, drop_path=0., dropout=0., gain=1., is_first=False):
        super().__init__()

        self.cross_attn = False
        if d_kv > 0:
            self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_q)
            self.encoder_decoder_attn = MultiHeadAttention(d_q, d_kv, d_q, d_q, num_heads, dropout, gain)
            self.cross_attn = True
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_q),
            linear(d_q, 4 * d_q, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_q, d_q, gain=gain),
            nn.Dropout(dropout))
        
        self.drop_path1 = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_layer_norm = nn.LayerNorm(d_q)
        self.self_attn = MultiHeadAttention(d_q, d_q, d_q // 4, d_q, num_heads, dropout, gain)
        self.conv = nn.Sequential(
            nn.Conv1d(d_q, d_q, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, input, encoder_output, N):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        B, L, D = input.shape
        if self.cross_attn:
            x = self.encoder_decoder_attn_layer_norm(input)
            x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
            x = self.conv(x.reshape(B*N, -1, D).transpose(1, 2)).transpose(1, 2).reshape(B, L, D)
            input = input + self.drop_path1(x)
            x = self.ffn(x)
            input = input + self.drop_path2(x)
        x = self.self_attn_layer_norm(input)
        x = x.reshape(B*N, -1, D)
        x = self.self_attn(x, x, x)
        x = input + x.reshape(B, -1, D)
        return x
    

class TransformerDecoder(nn.Module):
    
    def __init__(self, num_blocks, d_q, d_kv, num_heads, drop_path, dropout=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]  # stochastic depth decay rule
        gain = (3 * num_blocks) ** (-0.5)
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(d_q, d_kv, num_heads, dpr[0], dropout, gain, is_first=True)] +
            [TransformerDecoderBlock(d_q, d_kv, num_heads, dpr[i+1], dropout, gain, is_first=False)
                for i in range(num_blocks - 1)])   
        self.layer_norm = nn.LayerNorm(d_q)
    
    def forward(self, input, encoder_output, pos_emb, Nr):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for i, block in enumerate(self.blocks):
            input = block(input, encoder_output, Nr)
        return self.layer_norm(input)