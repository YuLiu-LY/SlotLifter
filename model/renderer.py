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
import torch.nn.functional as F

from model.nerf import NeRF
from model.slot_attn import JointDecoder, linear


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var

def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs)).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1],))
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class NeRFRenderer(nn.Module):

    def __init__(self, depth_range, cfg):
        super().__init__()
        self.depth_range = depth_range
        self.distance_scale = cfg.get('distance_scale', 25)
        self.weight_thres_color = 0.0001
        self.feat_weight_thres = 0.01
        self.alpha_mask_threshold = 0.0075
        self.step_size = None
        self.stop_semantic_grad = cfg.stop_semantic_grad
        self.nerf_mlp_dim = cfg.nerf_mlp_dim
        self.slot_dec_dim = cfg.slot_dec_dim
        self.n_samples = cfg.n_samples
        self.n_samples_fine = cfg.n_samples_fine
        self.num_instances = cfg.max_instances

        self.slot_size = cfg.slot_size
        self.inv_uniform = cfg.get('inv_uniform', False)
        self.normalize = cfg.normalize
        self.pos_proj = linear(120, self.slot_dec_dim)

    def compute_points_feature(self, nerf: NeRF, slot_dec: JointDecoder, 
                               xyz_sampled, slots, dists,
                               cam, src_imgs, src_cams, src_feats,
                               shape, is_train, rays_d, proj_mask=None):
        B, Nr, Np = shape
        points_feature, proj_mask = nerf.project_feature(xyz_sampled.view(B, Nr, Np, 3), cam, src_imgs, src_cams, src_feats, is_train, proj_mask)

        pos_emb = positional_encoding(xyz_sampled.view(B, Nr*Np, 3), 10)
        view_emb = positional_encoding(rays_d.view(B, Nr, 3), 10).unsqueeze(2).expand(-1, -1, Np, -1).reshape(B, Nr*Np, -1)
        pos_emb = self.pos_proj(torch.cat([pos_emb, view_emb], dim=-1))
        points_feature = points_feature + pos_emb

        points_coor = xyz_sampled.view(B, Nr*Np, 3)
        ret = slot_dec(points_feature, pos_emb, slots, points_coor, Nr)

        points_feature, w_slot_dec, sigma_slot = ret['x'], ret['w'], ret.get('sigma', None)
        points_feature = points_feature + pos_emb
        points_feature = points_feature.view(B*Nr, Np, self.slot_dec_dim)
        if sigma_slot is not None:
            sigma = sigma_slot.view(B*Nr, Np)
        else:
            sigma = nerf.compute_density(points_feature).view(B*Nr, Np) # [B*Nr, Np]
        alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale) 

        return points_feature, w_slot_dec, weight, ret.get('sparse_loss', torch.zeros(1, device=points_feature.device)), proj_mask, sigma
    
    def render_color(self, points_feature, viewdirs, w, appearance_mask, shape, color_mlp, white_bg, is_train):
        B, Nr, Np = shape
        rgb = points_feature.new_zeros((B*Nr, Np, 3))
        valid_rgbs = color_mlp(viewdirs[appearance_mask], points_feature[appearance_mask])
        rgb[appearance_mask] = valid_rgbs
        rgb_map = torch.sum(w * rgb, -2).reshape(B, Nr, 3)
        return rgb_map
    
    def render_instance(self, points_feature, w, appearance_mask, shape, w_slot_dec=None, instance_mlp=None):
        B, Nr, Np = shape
        if instance_mlp is not None:
            instances = points_feature.new_zeros((B*Nr, Np, self.num_instances))
            valid_instances = instance_mlp(points_feature[appearance_mask])
            instances[appearance_mask] = valid_instances
            instances = F.softmax(instances, dim=-1)
            instance_map = torch.sum(w * instances, -2).reshape(B, Nr, -1) # [B, Nr, K]
            instance_map = instance_map / (torch.sum(instance_map, -1, keepdim=True) + 1e-8)
        else:
            instances = w_slot_dec.view(B*Nr, Np, -1)[..., 1:]
            instance_map = torch.sum(w * instances, -2).reshape(B, Nr, -1) # [B, Nr, K]
            instance_map = instance_map / (torch.sum(instance_map, -1, keepdim=True) + 1e-6)

        return instance_map

    def render(self, nerf: NeRF, slot_dec: JointDecoder, 
               xyz_sampled, z_vals, viewdirs, slots, 
               cam, src_imgs, src_cams, src_feats,
               white_bg, is_train, 
               render_color=False, render_ins=False, render_depth=False, rays_d=None, proj_mask=None):
        B, Nr, Np, _ = viewdirs.shape
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        points_feature, w_slot_dec, weight, sparse_loss, proj_mask, sigma = self.compute_points_feature(nerf, slot_dec,
                                                                        xyz_sampled, slots, dists,
                                                                        cam, src_imgs, src_cams, src_feats,
                                                                        (B, Nr, Np), is_train, rays_d, proj_mask)
        appearance_mask = weight > self.weight_thres_color # [B*Nr, Np]
        w = weight[..., None] # [B*Nr, Np, 1]

        ret = {
            "proj_mask": proj_mask,
            "weight": weight.clone().detach(),
            "sparse_loss": sparse_loss,
            "w": w_slot_dec.view(B, Nr*Np, -1)[..., 1:],
            "pts": xyz_sampled.view(B, Nr*Np, 3),
        }

        if render_color:
            viewdirs = viewdirs.reshape(B*Nr, Np, 3)
            rgb_map = self.render_color(points_feature, viewdirs, w, appearance_mask, (B, Nr, Np), nerf.color_mlp, white_bg, is_train)
            ret["rgb"] = rgb_map
        
        if render_depth:
            depth_map = torch.sum(weight * z_vals, -1).reshape(B, Nr)
            opacity_map = torch.sum(w, -2).reshape(B, Nr)
            depth_map = depth_map + (1. - opacity_map) * z_vals.max()
            ret["depth"] = depth_map

        if render_ins:
            if self.stop_semantic_grad:
                w = w.detach()
            instance_map = self.render_instance(points_feature, w, appearance_mask, (B, Nr, Np), w_slot_dec)
            ret["instance"] = instance_map

        return ret

    def forward(self, nerf_c: NeRF, nerf_f: NeRF, slot_dec: JointDecoder, slot_dec_fine: JointDecoder, 
                rays, depth_range, slots,
                cam, src_imgs, src_cams, src_feats,
                white_bg=False, is_train=False, 
                render_color=False, render_depth=False,
                render_ins=False):
        r"""
        Input:
            rays: [B, Nr, 6]
            slots: [B, K, D]
            white_bg: True or False
            is_train: True or False
            cam: [B, 27]
            src_imgs: [B, Nv, 3, H, W]
            src_cams: [B, Nv, 27]
            src_feats: [B, Nv, D, H1, W1]
        Output:
            rgb: [B, Nr, 3]
            instance: [B, Nr, K]
            depth: [B, Nr]
            feats: [B, Nr, D]
        """
        B, Nr, _ = rays.shape
        # assert B == 1, "Only support batch size 1"
        Np = self.n_samples
        rays = rays.reshape(-1, 6)
        rays_o, rays_d = rays[:, :3], rays[:, 3:]
        xyz_sampled, z_vals = sample_points_in_box(rays_o, rays_d, Np, 
                                                   self.depth_range if self.depth_range is not None else depth_range, is_train, self.inv_uniform) # [B*Nr, n_samples, 3], [B*Nr, n_samples]
        viewdirs = rays_d.view(B, Nr, 1, 3).expand(-1, -1, Np, -1)

        ret_c = self.render(nerf_c, slot_dec, xyz_sampled, z_vals, viewdirs,
                            slots, cam, src_imgs, src_cams, src_feats, 
                            white_bg, is_train, 
                            render_color=render_color, render_ins=render_ins,
                            render_depth=render_depth, rays_d=rays_d)
        weight = ret_c["weight"]
        ret = {}
        for k, v in ret_c.items():
            if k != "weight" and k != 'proj_mask':
                ret[k + "_c"] = v

        if nerf_f is not None:
            Np_fine = self.n_samples_fine
            xyz_sampled, z_vals = sample_fine_pts(rays_o, rays_d, weight, z_vals, Np_fine, Np, is_train, self.inv_uniform)
            viewdirs = rays_d.view(B, Nr, 1, 3).expand(-1, -1, Np + Np_fine, -1)
            ret_f = self.render(nerf_f, slot_dec_fine, xyz_sampled, z_vals, viewdirs,
                                slots, cam, src_imgs, src_cams, src_feats, 
                                white_bg, is_train,
                                render_color=render_color, render_ins=render_ins,
                                render_depth=render_depth, rays_d=rays_d, proj_mask=ret_c['proj_mask'])
            for k,v in ret_f.items():
                if k != "weight" and k != 'proj_mask':
                    ret[k + "_f"] = v
        return ret
    
    @staticmethod
    def raw_to_alpha(sigma, dist):
        alpha = 1. - torch.exp(-sigma * dist)
        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
        weights = alpha * T[:, :-1]
        return alpha, weights, T[:, -1:]
    

def sample_points_in_box(rays_o, rays_d, n_samples, depth_range, is_train, inv_uniform=False):
    if isinstance(depth_range, tuple):
        depth_range = torch.tensor(depth_range).float().to(rays_o.device)
    depth_range = depth_range.expand(rays_d.shape[0], -1)
    near_depth, far_depth = depth_range[..., 0], depth_range[..., 1]
    if inv_uniform:
        start = 1.0 / near_depth  # [N_rays,]
        step = (1.0 / far_depth - start) / (n_samples - 1)
        inv_z_vals = torch.stack(
            [start + i * step for i in range(n_samples)], dim=1
        )  # [N_rays, n_samples]
        z_vals = 1.0 / inv_z_vals
        if is_train:
            mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
            # uniform samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand  # [N_rays, Np]
    else:
        step_size = (depth_range[..., 1:2] - depth_range[..., 0:1]) / (n_samples - 1)
        rng = torch.arange(n_samples)[None].type_as(rays_o).expand(rays_o.shape[:-1] + (n_samples,))
        if is_train:
            rng = rng + torch.rand_like(rng[:, [0]]).type_as(rng)
        step = step_size * rng.to(rays_o.device)
        z_vals = (depth_range[..., 0:1] + step) # [B*Nr, n_samples]

    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None] # [B*Nr, n_samples, 3]
    return rays_pts, z_vals


def sample_pdf(bins, weights, Np, det=False):
    r'''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param Np: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, Np]
    '''

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, Np, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)  # [N_rays, Np]
    else:
        u = torch.rand(bins.shape[0], Np, device=bins.device)
    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, Np]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, Np, 2]

    cdf = cdf.unsqueeze(1).repeat(1, Np, 1)  # [N_rays, Np, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, Np, 2]

    bins = bins.unsqueeze(1).repeat(1, Np, 1)  # [N_rays, Np, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, Np, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, Np]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, Np]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples

def sample_fine_pts(rays_o, rays_d, weights, z_vals, N_importance, Np, det=True, inv_uniform=False):
    if inv_uniform:
        inv_z_vals = 1.0 / z_vals
        inv_z_vals_mid = 0.5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])  # [N_rays, Np-1]
        weights = weights[:, 1:-1]  # [N_rays, Np-2]
        inv_z_vals = sample_pdf(
            bins=torch.flip(inv_z_vals_mid, dims=[1]),
            weights=torch.flip(weights, dims=[1]),
            Np=N_importance,
            det=det,
        )  # [N_rays, N_importance]
        z_samples = 1.0 / inv_z_vals
    else:
        # take mid-points of depth samples
        z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, Np-1]
        weights = weights[:, 1:-1]  # [N_rays, Np-2]
        z_samples = sample_pdf(
            bins=z_vals_mid, weights=weights, Np=N_importance, det=det
        )  # [N_rays, N_importance]

    z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, Np + N_importance]

    # samples are sorted with increasing depth
    z_vals, _ = torch.sort(z_vals, dim=-1)
    N_total_samples = Np + N_importance

    viewdirs = rays_d.unsqueeze(1).repeat(1, N_total_samples, 1)
    ray_o = rays_o.unsqueeze(1).repeat(1, N_total_samples, 1)
    pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, Np + N_importance, 3]
    return pts, z_vals
