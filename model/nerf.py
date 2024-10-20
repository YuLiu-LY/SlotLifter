# MIT License
#
# Copyright (c) 2022 Anpei Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from model.slot_attn import linear
from model.projection import Projector


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var


class NeRF(nn.Module):
    def __init__(self, cfg=None, n_samples=64):
        super().__init__()
        slot_dec_dim = cfg.slot_dec_dim        
        self.nerf_mlp_dim = cfg.nerf_mlp_dim
        self.color_mlp = RenderMLP(slot_dec_dim, 3, cfg.pe_view, cfg.pe_feat, self.nerf_mlp_dim, cfg.normalize)
        if not cfg.slot_density:
            self.density_proj = linear(slot_dec_dim, 1)

        self.projector = Projector()
        self.grid_init = cfg.grid_init
        self.random_proj_ratio = cfg.random_proj_ratio
        self.slot_dec_dim = slot_dec_dim
        if self.random_proj_ratio > 0:
            if cfg.num_src_view > 1:
                self.base_fc = linear(2 * (cfg.feature_size + 3), slot_dec_dim)
            else:
                self.base_fc = linear(cfg.feature_size + 3, slot_dec_dim)

        self.pos_encoding = self.posenc(slot_dec_dim, n_samples)
    
    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).float().unsqueeze(0)
        return sinusoid_table
        
    def init_one_svd(self, n_components, grid_resolution, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_components[i], grid_resolution[mat_id_1], grid_resolution[mat_id_0])), requires_grad=True))
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_components[i], grid_resolution[vec_id], 1)), requires_grad=True))
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)
    
    def get_coordinate_plane_line(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matrix_mode[0]], xyz_sampled[..., self.matrix_mode[1]], xyz_sampled[..., self.matrix_mode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vector_mode[0]], xyz_sampled[..., self.vector_mode[1]], xyz_sampled[..., self.vector_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line
    
    def compute_feature(self, xyz_sampled):
        if self.grid_init == 'tensorf':
            coordinate_plane, coordinate_line = self.get_coordinate_plane_line(xyz_sampled)
            plane_coef_point, line_coef_point = [], []
            for idx_plane in range(len(self.plane_grid)):
                plane_coef_point.append(F.grid_sample(self.plane_grid[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))
                line_coef_point.append(F.grid_sample(self.line_grid[idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
            return self.basis_mat((plane_coef_point * line_coef_point).T)
        elif self.grid_init == '3d':
            points_feature = F.grid_sample(self.grid, xyz_sampled.reshape(1, -1, 1, 1, 3), align_corners=True).squeeze().permute(1, 0)
            return self.basis_mat(points_feature)
    
    def project_feature(self, xyz_sampled, cam, src_imgs, src_cams, src_feats, is_Train=False, proj_mask=None):
        r"""
        Input:
            xyz_sampled: (B, Nr, Np, 3)
            cam: (B, 27)
            src_imgs: (B, Nv, 3, H, W)
            src_cams: (B, Nv, 27)
            src_feats: (B, Nv, D, H1, W1)
        Output:
            feature: (B, Nr*Np, D)
        """
        B, Nr, Np, _ = xyz_sampled.shape # number of points
        xyz_sampled = xyz_sampled.view(B, -1, 3) # (B, Nr*Np, 3)
        src_imgs = src_imgs.type_as(src_feats)
        points_feature = xyz_sampled.new_zeros(B, Nr*Np, self.slot_dec_dim)
        if self.random_proj_ratio > 0:
            if self.random_proj_ratio < 1 and is_Train:
                if proj_mask is None:
                    noise = torch.rand([1, Nr, 1, 1], device=xyz_sampled.device)
                    proj_mask = noise <= self.random_proj_ratio
                    m = proj_mask.expand(1, Nr, Np, 1).reshape(1, -1, 1)
                else:
                    m = proj_mask.expand(1, Nr, Np, 1).reshape(1, -1, 1)
                if torch.sum(m) > 0:
                    rgb_feat, mask, pixel_locations = self.projector.compute(xyz_sampled[m.expand(B, -1, 3)].view(B, -1, 3), cam, src_imgs, src_cams, src_feats)
                    Nv = src_imgs.shape[1]
                    if Nv == 1:
                        x = self.base_fc(rgb_feat)
                        points_feature[m.expand(B, -1, self.slot_dec_dim)] = torch.sum(x * mask, dim=2).view(-1)
                    else:
                        weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-6)
                        mean, var = fused_mean_variance(rgb_feat, weight)
                        x = torch.cat([mean, var], dim=-1).squeeze(2)
                        points_feature[m.expand(B, -1, self.slot_dec_dim)] = self.base_fc(x).view(-1)
                else:
                    mask = torch.zeros(B, Nr*Np, 1, 1, device=xyz_sampled.device)
            else:
                rgb_feat, mask, pixel_locations = self.projector.compute(xyz_sampled.view(B, -1, 3), cam, src_imgs, src_cams, src_feats)
                Nv = src_imgs.shape[1]
                if Nv == 1:
                    points_feature = torch.sum(self.base_fc(rgb_feat) * mask, dim=2)
                else:
                    weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-6)
                    mean, var = fused_mean_variance(rgb_feat, weight)
                    x = torch.cat([mean, var], dim=-1).squeeze(2)
                    points_feature = self.base_fc(x)
        return points_feature, proj_mask
    
    def compute_density(self, points_feature):
        sigma = F.relu(self.density_proj(points_feature).squeeze(-1)) # (B, N)
        return sigma


class RenderMLP(nn.Module):

    def __init__(self, in_channels, out_channels=3, pe_view=2, pe_feat=2, nerf_mlp_dim=128, normalize=True):
        super().__init__()
        self.pe_view = pe_view
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.view_independent = self.pe_view == 0 and self.pe_feat == 0
        self.in_feat_mlp = in_channels

        self.mlp = nn.Sequential(
            linear(self.in_feat_mlp, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'), 
            nn.LeakyReLU(), 
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, out_channels)
        )
        self.normalize = normalize

    def forward(self, rays_d, features):
        out = self.mlp(features)

        if self.normalize:
            out = out.tanh()
        else:
            out = out.tanh() / 2 + 0.5
        return out


class SemanticMLP(nn.Module):
    def __init__(self, in_channels, out_channels=3, pe_feat=2, nerf_mlp_dim=128):
        super().__init__()
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.in_feat_mlp = 2 * pe_feat * in_channels + in_channels

        self.mlp = nn.Sequential(
            linear(self.in_feat_mlp, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'), 
            nn.LeakyReLU(), 
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, out_channels)
        )

    def forward(self, features):
        indata = [features]
        if self.pe_feat > 0:
            indata += [positional_encoding(features, self.pe_feat)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        return out
    

class InstanceMLP(nn.Module):
    def __init__(self, in_channels, out_channels=3, pe_feat=2, nerf_mlp_dim=128):
        super().__init__()
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.in_feat_mlp = 2 * pe_feat * in_channels + in_channels

        self.mlp = nn.Sequential(
            linear(self.in_feat_mlp, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'), 
            nn.LeakyReLU(), 
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, nerf_mlp_dim, weight_init='kaiming', nonlinearity='leaky_relu'),
            nn.LeakyReLU(),
            linear(nerf_mlp_dim, out_channels)
        )

    def forward(self, features):
        indata = [features]
        if self.pe_feat > 0:
            indata += [positional_encoding(features, self.pe_feat)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        return out


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs)).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1],))
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

