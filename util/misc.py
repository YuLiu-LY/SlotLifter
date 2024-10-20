# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from collections import OrderedDict
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import torch
import math
import numpy as np
from pathlib import Path
from matplotlib import cm
from PIL import Image
import torchvision.transforms as T


def visualize_depth(depth, minval=0.001, maxval=1.5, use_global_norm=True):
    x = depth
    if isinstance(depth, torch.Tensor):
        x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    if use_global_norm:
        mi = minval
        ma = maxval
    else:
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x_ = Image.fromarray((cm.get_cmap('jet')(x) * 255).astype(np.uint8))
    x_ = T.ToTensor()(x_)[:3, :, :]
    return x_


def bounds(x):
    lower = []
    upper = []
    for i in range(x.shape[1]):
        lower.append(x[:, i].min())
        upper.append(x[:, i].max())
    return torch.tensor([lower, upper])


def visualize_points(points, vis_path, colors=None):
    if colors is None:
        Path(vis_path).write_text("\n".join(f"v {p[0]} {p[1]} {p[2]} 127 127 127" for p in points))
    else:
        Path(vis_path).write_text("\n".join(f"v {p[0]} {p[1]} {p[2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}" for i, p in enumerate(points)))


def visualize_points_as_pts(points, vis_path, colors=None):
    if colors is None:
        Path(vis_path).write_text("\n".join([f'{points.shape[0]}'] + [f"{p[0]} {p[1]} {p[2]} 255 127 127 127" for p in points]))
    else:
        Path(vis_path).write_text("\n".join([f'{points.shape[0]}'] + [f"{p[0]} {p[1]} {p[2]} 255 {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}" for i, p in enumerate(points)]))


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen if t.requires_grad]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s)], axis=1)


def get_parameters_from_state_dict(state_dict, filter_key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(filter_key):
            new_state_dict[k.replace(filter_key + '.', '')] = state_dict[k]
    return new_state_dict


def logistic(n, zero_at):
    return 1 - 1 / (1 + math.exp(-10 * (n / zero_at - 0.5)))


def visualize_voxel_grid(output_path, voxel_grid, scale_to=(-1, 1)):
    voxel_grid = ((voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min())).cpu()
    rescale = lambda axis: scale_to[0] + (points[axis] / voxel_grid.shape[axis]) * (scale_to[1] - scale_to[0])
    points = list(torch.where(voxel_grid > 0))
    if len(points[0] > 0):
        colors = cm.get_cmap('jet')(voxel_grid.numpy())
        colors = colors[points[0].numpy(), points[1].numpy(), points[2].numpy(), :]
        points[0] = rescale(0)
        points[1] = rescale(1)
        points[2] = rescale(2)
        Path(output_path).write_text("\n".join([f'v {points[0][i]} {points[1][i]} {points[2][i]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}' for i in range(points[0].shape[0])]))
    else:
        Path(output_path).write_text("")
        print("no points found..")


def visualize_labeled_points(locations, labels, output_path):
    from util.distinct_colors import DistinctColors
    distinct_colors = DistinctColors()
    if isinstance(labels, torch.Tensor):
        colored_arr = distinct_colors.get_color_fast_torch(labels.flatten().cpu().numpy().tolist()).reshape(list(labels.shape) + [3]).numpy()
    else:
        colored_arr = distinct_colors.get_color_fast_numpy(labels.flatten().tolist()).reshape(list(labels.shape) + [3])
    visualize_points(locations, output_path, colored_arr)


def visualize_weighted_points(output_path, xyz, weights, threshold=1e-4):
    weights = weights.view(-1)
    weights_mask = weights > threshold
    colors = cm.get_cmap('jet')(weights[weights_mask].numpy())
    visualize_points(xyz[weights_mask, :].numpy(), output_path, colors=colors)


def visualize_mask(arr, path):
    from util.distinct_colors import DistinctColors
    distinct_colors = DistinctColors()
    assert len(arr.shape) == 2, "should be an HxW array"
    boundaries = get_boundary_mask(arr)
    if isinstance(arr, torch.Tensor):
        colored_arr = distinct_colors.get_color_fast_torch(arr.flatten().cpu().numpy().tolist()).reshape(list(arr.shape) + [3]).numpy()
    else:
        colored_arr = distinct_colors.get_color_fast_numpy(arr.flatten().tolist()).reshape(list(arr.shape) + [3])
    colored_arr = (colored_arr * 255).astype(np.uint8)
    colored_arr[boundaries > 0, :] = 0
    Image.fromarray(colored_arr).save(path)


def probability_to_normalized_entropy(probabilities):
    entropy = torch.zeros_like(probabilities[:, 0])
    for i in range(probabilities.shape[1]):
        entropy = entropy - probabilities[:, i] * torch.log2(probabilities[:, i] + 1e-8)
    entropy = entropy / math.log2(probabilities.shape[1])
    return entropy


def get_boundary_mask(arr, dialation_size=1):
    import cv2
    arr_t, arr_r, arr_b, arr_l = arr[1:, :], arr[:, 1:], arr[:-1, :], arr[:, :-1]
    arr_t_1, arr_r_1, arr_b_1, arr_l_1 = arr[2:, :], arr[:, 2:], arr[:-2, :], arr[:, :-2]
    kernel = np.ones((dialation_size, dialation_size), 'uint8')
    if isinstance(arr, torch.Tensor):
        arr_t = torch.cat([arr_t, arr[-1, :].unsqueeze(0)], dim=0)
        arr_r = torch.cat([arr_r, arr[:, -1].unsqueeze(1)], dim=1)
        arr_b = torch.cat([arr[0, :].unsqueeze(0), arr_b], dim=0)
        arr_l = torch.cat([arr[:, 0].unsqueeze(1), arr_l], dim=1)

        arr_t_1 = torch.cat([arr_t_1, arr[-2, :].unsqueeze(0), arr[-1, :].unsqueeze(0)], dim=0)
        arr_r_1 = torch.cat([arr_r_1, arr[:, -2].unsqueeze(1), arr[:, -1].unsqueeze(1)], dim=1)
        arr_b_1 = torch.cat([arr[0, :].unsqueeze(0), arr[1, :].unsqueeze(0), arr_b_1], dim=0)
        arr_l_1 = torch.cat([arr[:, 0].unsqueeze(1), arr[:, 1].unsqueeze(1), arr_l_1], dim=1)

        boundaries = torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_and(arr_t != arr, arr_t_1 != arr), torch.logical_and(arr_r != arr, arr_r_1 != arr)), torch.logical_and(arr_b != arr, arr_b_1 != arr)), torch.logical_and(arr_l != arr, arr_l_1 != arr))

        boundaries = boundaries.cpu().numpy().astype(np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)
        boundaries = torch.from_numpy(boundaries).to(arr.device)
    else:
        arr_t = np.concatenate([arr_t, arr[-1, :][np.newaxis, :]], axis=0)
        arr_r = np.concatenate([arr_r, arr[:, -1][:, np.newaxis]], axis=1)
        arr_b = np.concatenate([arr[0, :][np.newaxis, :], arr_b], axis=0)
        arr_l = np.concatenate([arr[:, 0][:, np.newaxis], arr_l], axis=1)

        arr_t_1 = np.concatenate([arr_t_1, arr[-2, :][np.newaxis, :], arr[-1, :][np.newaxis, :]], axis=0)
        arr_r_1 = np.concatenate([arr_r_1, arr[:, -2][:, np.newaxis], arr[:, -1][:, np.newaxis]], axis=1)
        arr_b_1 = np.concatenate([arr[0, :][np.newaxis, :], arr[1, :][np.newaxis, :], arr_b_1], axis=0)
        arr_l_1 = np.concatenate([arr[:, 0][:, np.newaxis], arr[:, 1][:, np.newaxis], arr_l_1], axis=1)

        boundaries = np.logical_or(np.logical_or(np.logical_or(np.logical_and(arr_t != arr, arr_t_1 != arr), np.logical_and(arr_r != arr, arr_r_1 != arr)), np.logical_and(arr_b != arr, arr_b_1 != arr)), np.logical_and(arr_l != arr, arr_l_1 != arr)).astype(np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)

    return boundaries


def pixelid2patchid(pixelid, H, W, H1, W1):
    """
    Input:
        pixelid: from 0 to H*W-1
    Return:
        patchid: from 0 to H1*W1-1
    """
    x, y = pixelid % W, pixelid // W
    x1, y1 = x // (W // W1), y // (H // H1)
    patchid = y1 * W1 + x1
    return patchid


def patchify(x, p):
    r"""
    Input:
        x: (H, W, D)
        p: patch size
    Return:
        x: (hw, p*p, D)
    """
    H, W, D = x.shape
    h, w = H // p, W // p
    x = x.reshape(h, p, w, p, D)
    x = torch.einsum('hpwqc->hwpqc', x)
    x = x.reshape(h*w, p*p, D)
    return x


def batch_patchify(x, p):
    r"""
    Input:
        x: (B, H, W, D)
        p: patch size
    Return:
        x: (B, hw, p*p, D)
    """
    B, H, W, D = x.shape
    h, w = H // p, W // p
    x = x.reshape(B, h, p, w, p, D)
    x = torch.einsum('bhpwqc->bhwpqc', x)
    x = x.reshape(B, h*w, p*p, D)
    return x


class SubSampler():
    def idx_subsample(self, img_size, Br, mode='random'):
        r'''
        Input:
            img_size: H,W
            Br: number of rays to sample
        Return:
            subsample_idx: [Br, 1]
        '''
        if mode == 'uniform':
            b = int(Br ** 0.5)
            assert b * b == Br, "Br must be a square number"
            H, W = img_size
            Nr = H * W
            s_x, s_y = W // b, H // b
            x = torch.randint(0, s_x, size=(1,)).item()
            y = torch.randint(0, s_y, size=(1,)).item()
            subsample_idx = torch.arange(Nr).reshape(H, W)[y:, x:][::s_y, ::s_x]
            subsample_idx = subsample_idx.reshape(-1, 1)
            Br1 = subsample_idx.shape[0]
            if Br1 < Br: # b can't be devided by H or W
                subsample_idx = torch.cat([subsample_idx, 
                                        torch.randint(0, Nr, [Br-Br1, 1])], dim=0)
            else:
                subsample_idx = subsample_idx[torch.randperm(Br1)[:Br]]
            return subsample_idx
        else:
            H, W = img_size
            Nr = H * W
            subsample_idx = torch.randperm(Nr)[:Br].reshape(-1, 1)
            return subsample_idx
    
    def idx_subsample_patch(self, img_size, patch_size, s=1):
        r'''
        Input:
            img_size: H,W
            patch_size: P
            s: stride for subsampling
        Return:
            subsample_idx: [Br, 1]
        '''
        H, W = img_size
        Nr = H * W
        P = patch_size
        x = torch.randint(0, W//P, size=(1,)).item() * P + torch.randint(0, s, size=(1,)).item()
        y = torch.randint(0, H//P, size=(1,)).item() * P + torch.randint(0, s, size=(1,)).item()
        subsample_idx = torch.arange(Nr).view(H, W)[y:y+P, x:x+P][::s, ::s]
        subsample_idx = subsample_idx.reshape(P//s*P//s, 1)
        return subsample_idx
    
    def subsample(self, idx, x_tuple):
        r'''
        Input:
            idx: [Br, 1]
            x_tuple: a tuple of tensors to subsample, [HW, C]
        Return:
            x_tuple: a tuple of tensors [Br, C]
        '''
        ret = []
        for x in x_tuple:
            ret.append(x.gather(0, idx.expand(-1, x.shape[1])))    
        return tuple(ret)
    
