import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from model.transformer import TransformerDecoder
from model.encoder import MultiViewResUNet


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1., nonlinearity='relu'):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m


class SlotAttention(nn.Module):
    def __init__(
        self,
        feature_size,
        slot_size, 
        drop_path=0.2,
        num_head=1,
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = 1.0
        self.num_head = num_head

        self.norm_feature = nn.LayerNorm(feature_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(feature_size, slot_size, bias=False)
        self.project_v = linear(feature_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, slot_size * 4, weight_init='kaiming'),
            nn.ReLU(),
            linear(slot_size * 4, slot_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, features, slots_init, num_src_view, num_iter=3):
        # features: [batch_size, num_feature, inputs_size]
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features) 

        slots = slots_init
        # Multiple rounds of attention.
        for i in range(num_iter - 1):
            slots, attn = self.iter(slots, k, v, num_src_view=num_src_view)
            slots = slots.detach() + slots_init - slots_init.detach()
        slots, attn = self.iter(slots, k, v, num_src_view=num_src_view)
        return slots, attn

    def iter(self, slots, k, v, num_src_view):
        B, K, D = slots.shape
        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.project_q(slots)

        Nh = self.num_head
        q = q.reshape(B, K, Nh, D//Nh).transpose(1, 2) # [B, Nh, K, D//Nh]
        k = k.reshape(B, -1, Nh, D//Nh).transpose(1, 2) # [B, Nh, Nf, D//Nh]

        # Attention
        scale = (D//Nh) ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale # [B, Nh, K, Nf]
        attn_logits = attn_logits.mean(1) # [B, K, Nf]
        attn = F.softmax(attn_logits, dim=1)

        # # Weighted mean
        attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
        attn_wm = attn / attn_sum 
        updates = torch.einsum('bij, bjd->bid', attn_wm, v)            

        # Update slots
        slots = self.gru(
            updates.reshape(-1, D),
            slots_prev.reshape(-1, D)
        )
        slots = slots.reshape(B, -1, D)
        slots = slots + self.drop_path(self.mlp(self.norm_mlp(slots)))
        return slots, attn


class SlotEnc(nn.Module):
    def __init__(
        self, num_iter, num_slots, feature_size,
        slot_size, drop_path=0.2, num_blocks=1):
        super().__init__()
        
        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.slot_attn = self.slot_attn = nn.ModuleList([
                SlotAttention(feature_size, slot_size, drop_path=drop_path) for i in range(num_blocks)
            ])
        self.num_blocks = num_blocks
        self.slots_init = nn.Parameter(torch.zeros(1, num_slots, slot_size))
        nn.init.xavier_uniform_(self.slots_init)

    def forward(self, f, sigma, num_src_view):
        B, _, D = f.shape
        # initialize slots.
        mu = self.slots_init.expand(B, -1, -1)
        z = torch.randn_like(mu).type_as(f)
        slots = mu + z * sigma * mu.detach()
        slots, attn = self.slot_attn[0](f, slots, num_iter=self.num_iter, num_src_view=num_src_view)
        for i in range(self.num_blocks - 1):
            slots, attn = self.slot_attn[i + 1](f, slots, num_iter=self.num_iter, num_src_view=num_src_view)
        return slots, attn


class Slot3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_size = config.feature_size
        self.multi_view_enc = MultiViewResUNet(config)
        self.slot_enc = SlotEnc(
            num_iter=config.num_iter,
            num_slots=config.num_slots,
            feature_size=config.feature_size,
            slot_size=config.slot_size,
            drop_path=config.drop_path,
        )
        self.num_slots = config.num_slots

    def forward(self, src_cams=None, images=None, sigma=0):
        """
        Input:
            images: [batch_size, N_views, 3, H, W]
            src_cams: [batch_size, N_views, 25]
        Output:
            slots: [batch_size, num_slots, slot_size]
        """
        B, N_view = src_cams.shape[:2]
        H, W = images.shape[3:]
        features = self.multi_view_enc(src_cams, images.reshape(B*N_view, 3, H, W)) # [B*N_views, D, H1, W1]
        
        H1, W1 = features.shape[-2:]
        features = features.permute(0, 2, 3, 1) # [B*N_views, H1, W1, D]

        feats = features.reshape(B, -1, self.feature_size)
        attn = torch.zeros(B, self.num_slots, feats.shape[1], device=feats.device)
        slots, attn = self.slot_enc(feats, sigma=sigma, num_src_view=N_view) # [B, K, slot_size]
        return slots, attn, features.reshape(B, N_view, H1, W1, features.shape[-1])


class JointDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_kv = cfg.slot_size
        self.num_slots = cfg.num_slots
        self.tf = TransformerDecoder(
            cfg.num_dec_blocks, cfg.slot_dec_dim, d_kv, 4, cfg.drop_path, 0)
        self.empty_slot = nn.Parameter(torch.zeros(1, 1, cfg.slot_size))
        nn.init.xavier_uniform_(self.empty_slot)

        self.Wq = linear(cfg.slot_dec_dim, cfg.slot_dec_dim)
        self.Wk = linear(cfg.slot_size, cfg.slot_dec_dim)
        self.out_proj = linear(cfg.slot_size, cfg.slot_dec_dim)
        self.force_bg = False
        self.bg_bound = cfg.bg_bound
        self.scale = cfg.slot_dec_dim ** -0.5
        self.slot_density = cfg.get('slot_density', False)
        if self.slot_density:
            self.density_scale = nn.Parameter(torch.zeros(1))


    def forward(self, point_feats, points_emb, slots, points_coor, Nr=0):
        r"""
        Input:
            point_feats: [B, N, D]  N = N_ray * N_points
            slots: [B, K, slot_size]
        Output:
            x: [B, N, D]
            w: [B, N, K]
        """ 
        slots = torch.cat([self.empty_slot.expand(slots.shape[0], -1, -1), slots], dim=1) # [B, K+1, D]
        x = self.tf(point_feats, slots, points_emb, Nr) # [B, N, D]

        # point slot mapping
        q = self.Wq(x)
        k = self.Wk(slots)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [B, N, K+1]
        if self.force_bg:
            out_idx = (points_coor.abs() > self.bg_bound).any(-1)[..., None].repeat(1, 1, self.num_slots+1) # [B, N]
            out_idx[:, :, 0:2] = False
            logits[out_idx] = -torch.inf
        w = F.softmax(logits, dim=-1) # [B, N, K+1]
        x = torch.matmul(w, slots) # [B, N, D]
        x = self.out_proj(x)
        if self.slot_density:
            slot_sigma = F.relu(logits)
            sigma = (slot_sigma[..., 1:] * w[..., 1:]).sum(-1)
            sigma = sigma * self.density_scale.exp() # [B, N]
        else:
            sigma = None
            
        return {'x': x, 'w': w, 'sigma': sigma}
        

