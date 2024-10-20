# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch


def create_grid(H, W, render_stride=1):
    xs = torch.linspace(0, W - 1, W)[::render_stride]
    ys = torch.linspace(0, H - 1, H)[::render_stride]
    i, j = torch.meshgrid(xs, ys, indexing='ij')
    i, j = i + render_stride // 2, j + render_stride // 2
    return i.t(), j.t()


def get_ray_directions_with_intrinsics(H, W, intrinsics, render_stride=1):
    B = intrinsics.shape[0]
    h, w = H // render_stride, W // render_stride
    i, j = create_grid(H, W, render_stride=render_stride)
    i, j = i.to(intrinsics.device), j.to(intrinsics.device)
    i, j = i[None, ...], j[None, ...]
    fx, fy, cx, cy = intrinsics[:, 0:1, 0:1], intrinsics[:, 1:2, 1:2], intrinsics[:, 0:1, 2:3], intrinsics[:, 1:2, 2:3]
    directions = torch.stack([
        (i - cx) / fx, (j - cy) / fy, torch.ones([B, h, w], device=intrinsics.device)
    ], -1)
    return directions


def get_rays(cameras, H, W, render_stride=1):
    # cameras: (B, 27) 2 + 9 + 16  HW + intrinsics + cam2world
    h, w = H // render_stride, W // render_stride
    rays_o, rays_d = get_rays_origin_and_direction(cameras, H, W, render_stride) # (B, 1, 3), (B, H*W, 3)
    rays = torch.cat([
        rays_o.expand(-1, h*w, -1), rays_d
    ], -1) # (B, H*W, 6)
    return rays


def get_rays_origin_and_direction(cameras, H, W, render_stride=1):
    B = cameras.shape[0]
    h, w = H // render_stride, W // render_stride
    intrinsics, cam2world = cameras[:, 2:18].reshape(-1, 4, 4)[:, :3, :3], cameras[:, -16:].reshape(-1, 4, 4)
    directions = get_ray_directions_with_intrinsics(H, W, intrinsics, render_stride)
    # directions: (B, H, W, 3), cam2world: (B, 4, 4)
    rays_d = torch.matmul(directions.view(B, h*w, 3), cam2world[:, :3, :3].transpose(1, 2)) # (B, H*W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = cam2world[:, :3, 3]

    rays_d = rays_d.view(B, h*w, 3)
    rays_o = rays_o.view(B, 1, 3)

    return rays_o, rays_d


def rays_intersect_sphere(rays_o, rays_d, r=1):
    """
    Solve for t such that a=ro+trd with ||a||=r
    Quad -> r^2 = ||ro||^2 + 2t (ro.rd) + t^2||rd||^2
    -> t = (-b +- sqrt(b^2 - 4ac))/(2a) with
       a = ||rd||^2
       b = 2(ro.rd)
       c = ||ro||^2 - r^2
       => (forward intersection) t= (sqrt(D) - (ro.rd))/||rd||^2
       with D = (ro.rd)^2 - (r^2 - ||ro||^2)
    """
    odotd = torch.sum(rays_o * rays_d, -1)
    d_norm_sq = torch.sum(rays_d ** 2, -1)
    o_norm_sq = torch.sum(rays_o ** 2, -1)
    determinant = odotd ** 2 + (r ** 2 - o_norm_sq) * d_norm_sq
    assert torch.all(
        determinant >= 0
    ), "Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!"
    return (torch.sqrt(determinant) - odotd) / d_norm_sq
