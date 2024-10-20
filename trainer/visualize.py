# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm
from pathlib import Path
import torch
import hydra
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from dataset import get_dataset
from model.nerf import NeRF
from model.renderer import NeRFRenderer
from model.slot_attn import Slot3D
from model.slot_attn import SlotMixerDecoder
from util.misc import visualize_depth
from PIL import Image
import numpy as np
import seaborn as sns


def segmentation_to_rgb(seg, palette=None, num_objects=None, bg_color=(0, 0, 0)):
    seg = seg[..., None]
    if num_objects is None:
        num_objects = np.max(seg)  # assume consecutive numbering
    num_objects += 1  # background
    if palette is None:
        # palette = [bg_color] + sns.color_palette('hls', num_objects-1)
        palette = sns.color_palette('hls', num_objects)

    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.float32)
    for i in range(num_objects):
        seg_img[np.nonzero(seg[:, :, 0] == i)] = palette[i]
    return seg_img


def save_img_from_tensor(img, path, transform=False):
    r'''
    img: tensor, [H, W, 3]
    '''
    img = img.cpu().numpy() * 255
    img = img.astype(np.uint8)
    # if transform:
        # brighten the image 
        # img = cv2.convertScaleAbs(img, alpha=1.8, beta=1)
    img = Image.fromarray(img)
    img.save(path)


def save_seg_from_tensor(seg, path):
    r'''
    seg: tensor, [H, W]
    '''
    seg = seg.cpu().numpy()
    seg = segmentation_to_rgb(seg)
    seg = (seg * 255).astype(np.uint8)
    seg = Image.fromarray(seg)
    seg.save(path)


class Visualizer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.train_set, self.val_set, self.test_set = get_dataset(config)

        self.slot_3D = Slot3D(config)
        self.slot_dec = SlotMixerDecoder(config)
        self.slot_dec_fine = SlotMixerDecoder(config) if config.coarse_to_fine else None        
        self.nerf = NeRF(config, n_samples=config.n_samples)
        self.nerf_fine = NeRF(config, n_samples=config.n_samples+config.n_samples_fine) if config.coarse_to_fine else None
        self.depth_range = min(self.train_set.depth_range[0], self.val_set.depth_range[0], self.test_set.depth_range[0]), \
                    max(self.train_set.depth_range[1], self.val_set.depth_range[1], self.test_set.depth_range[1])
        self.renderer = NeRFRenderer(self.depth_range, cfg=config)

        self.cfg = config
        self.output_dir_result_images = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/images')
        self.output_dir_result_images.mkdir(exist_ok=True)
        self.output_dir_result_seg = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/seg')
        self.output_dir_result_seg.mkdir(exist_ok=True)
        self.output_dir_result_depth = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/depth')
        self.output_dir_result_depth.mkdir(exist_ok=True)
        self.output_dir_result_attn = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/attn')
        self.output_dir_result_attn.mkdir(exist_ok=True)

        self.scene_id = config.get('scene_id', 0)
        
    def forward(self, rays, depth_range, slots, view_feats, cam, src_rgbs, src_cams, is_train):
        B, Nr, _ = rays.shape
        outputs = []
        render_depth = True
        render_ins = True
        render_feat = False
        render_color = True
        for i in range(0, Nr, self.cfg.chunk):
            outputs.append(self.renderer(self.nerf, self.nerf_fine, self.slot_dec, self.slot_dec_fine, 
                                         rays[:, i: i + self.cfg.chunk], depth_range, slots, 
                                         cam, src_rgbs, src_cams, view_feats,
                                         False, is_train, 
                                         render_color=render_color, render_depth=render_depth,
                                         render_sem=False, render_ins=render_ins, render_feat=render_feat))
        keys = outputs[0].keys()
        out = {}
        for k in keys:
            if 'dist' in k or 'loss' in k:
                out[k] = torch.stack([o[k] for o in outputs], 0).mean()
            else:
                out[k] = torch.cat([o[k] for o in outputs], 1).flatten(0, 1)
        return out
        
    def visualize(self, dataloader):
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            if batch_idx == self.scene_id or self.cfg.dataset == 'scannet' or self.cfg.dataset == 'dtu':
                src_rgbs, src_cams = batch['src_rgbs'].to(self.device), batch['src_cams'].to(self.device)
                B, Nv, H, W, _ = src_rgbs.shape
                src_rgbs = src_rgbs.permute(0, 1, 4, 2, 3) # [B, Nv, 3, H, W]
                slots, attn, view_feats = self.slot_3D(None, sigma=0, images=src_rgbs, src_cams=src_cams) 
                view_feats = view_feats.permute(0, 1, 4, 2, 3) # [B, Nv, D, H, W]

                images = src_rgbs[0].reshape(-1, 3, H, W).cpu() # [N_src_view, 3, H, W]
                H1, W1 = H // 4, W // 4
                if self.cfg.dataset == 'dtu':
                    H1 = 76
                attn = attn.reshape(-1, src_rgbs.shape[1], H1, W1)
                attn = F.interpolate(attn, size=(H, W), mode='nearest')
                attn = attn.permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 3, 1, 1)
                img = torch.cat([images.unsqueeze(1).cpu(), 1 - attn.cpu()], dim=1).reshape(-1, 3, H, W)
                img = make_grid(img, nrow=img.shape[0], padding=0) # [3, (N_src_view+1)*H, W]
                save_img_from_tensor(img.permute(1, 2, 0), self.output_dir_result_attn / f"{batch_idx:04d}_attn.png")
                # input_img = images[0].permute(1, 2, 0) # [H, W, 3]
                # save_img_from_tensor(input_img, self.output_dir_result_images / f"{batch_idx:04d}_input.png", True)
                # instances = batch['instances'][0].view(H, W) # [H, W]
                # save_seg_from_tensor(instances, self.output_dir_result_seg / f"{batch_idx:04d}_seg_gt.png")
                if self.cfg.dataset == 'scannet':
                    gt_img = batch['rgbs'].view(H, W, 3) / 2 + 0.5 # [H, W, 3]
                    save_img_from_tensor(gt_img, self.output_dir_result_images / f"{batch_idx:04d}_rgb_gt.png", True)
                    gt_seg = batch['instances'].view(H, W) # [H, W]
                    save_seg_from_tensor(gt_seg, self.output_dir_result_seg / f"{batch_idx:04d}_seg_gt.png")

                depth_range = batch['depth_range'].to(self.device) # [B, 2]
                N = self.cfg.num_vis
                all_rays = batch['rays'][0].to(self.device)  # [N, HW, 6]
                for n in tqdm(range(N)):
                    rays = all_rays[n:n+1]# [1, HW, 6]
                    output = self(rays, depth_range, slots, view_feats, batch.get('azi_rot'), src_rgbs, src_cams, False)
                
                    if self.cfg.coarse_to_fine:
                        output_rgb = output['rgb_f']
                        output_instances = output['instance_f']
                        output_depth = output['depth_f']
                    else:
                        output_rgb = output['rgb_c']
                        output_instances = output['instance_c']
                        output_depth = output['depth_c']

                    if self.cfg.normalize:
                        output_rgb = output_rgb * 0.5 + 0.5
                        src_rgbs = src_rgbs * 0.5 + 0.5

                    shape = (H, W, 3)
                    imgs_pred = output_rgb.view(shape)
                    seg_pred = output_instances.argmax(-1).view(H, W)
                    depth_pred = output_depth.view(H, W)
                    save_img_from_tensor(imgs_pred, self.output_dir_result_images / f"{batch_idx:04d}_{n:02d}_rgb_pred.png", True)
                    save_seg_from_tensor(seg_pred, self.output_dir_result_seg / f"{batch_idx:04d}_{n:02d}_seg_pred.png")
                    depth = visualize_depth(depth_pred, maxval=self.depth_range[1], use_global_norm=True) # [3, H, W]
                    save_img_from_tensor(depth.permute(1, 2, 0), self.output_dir_result_depth / f"{batch_idx:04d}_{n:02d}_depth_pred.png")
                # break

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)


@hydra.main(config_path='../config', config_name='config', version_base='1.2')
def main(config):
    config = config.cfg 
    result_path = Path(f'{config.log_path}/{config.exp_name}')
    result_path.mkdir(exist_ok=True)
    model = Visualizer(config)
    ckpt = torch.load(config.ckpt_path) 
    model.load_state_dict(ckpt['state_dict'])
    print(f'Load from checkpoint: {config.ckpt_path}')
    model.cuda()
    model.eval()
    with torch.no_grad():
        model.visualize(model.val_dataloader())

if __name__ == '__main__':
    main()
