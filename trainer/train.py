# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
from pathlib import Path
import torch
import hydra
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from tabulate import tabulate
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from dataset import get_dataset
from model.nerf import NeRF
from model.renderer import NeRFRenderer
from trainer import create_trainer, visualize_panoptic_outputs
from util.misc import visualize_depth
from util.metrics import SegMetrics, ReconMetrics
from model.slot_attn import Slot3D
from torch import optim
from util.optimizer import Lion
from model.slot_attn import JointDecoder
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


def save_img_from_tensor(img, path):
    r'''
    img: tensor, [H, W, 3]
    '''
    img = img.cpu().numpy()
    img = (img * 255).astype(np.uint8)
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


class TensoRFTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.train_set, self.val_set, self.test_set = get_dataset(config)
        self.train_sampler = None
        if config.visualized_indices is None:
            if config.job_type == 'vis':
                config.visualized_indices = torch.arange(len(self.val_set)).tolist()
            else:
                config.visualized_indices = torch.randperm(len(self.val_set))[:4].tolist()
            if config.job_type == 'debug':
                config.visualized_indices = config.visualized_indices[0:1]
        self.instance_steps = config.instance_steps
        self.save_hyperparameters(config)

        self.slot_3D = Slot3D(config)
        self.slot_dec = JointDecoder(config)
        self.slot_dec_fine = JointDecoder(config) if config.coarse_to_fine else None        
        self.nerf = NeRF(config, n_samples=config.n_samples)
        self.nerf_fine = NeRF(config, n_samples=config.n_samples+config.n_samples_fine) if config.coarse_to_fine else None
        depth_range = min(self.train_set.depth_range[0], self.val_set.depth_range[0], self.test_set.depth_range[0]), \
                    max(self.train_set.depth_range[1], self.val_set.depth_range[1], self.test_set.depth_range[1])
        self.renderer = NeRFRenderer(depth_range, cfg=config)

        self.loss = torch.nn.MSELoss(reduction='mean')

        self.cfg = config
        self.output_dir_result_images = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/images')
        self.output_dir_result_images.mkdir(exist_ok=True)
        self.output_dir_result_seg = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/seg')
        self.output_dir_result_seg.mkdir(exist_ok=True)
        self.output_dir_result_depth = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/depth')
        self.output_dir_result_depth.mkdir(exist_ok=True)
        self.output_dir_result_attn = Path(f'{self.cfg.log_path}/{self.cfg.exp_name}/attn')
        self.output_dir_result_attn.mkdir(exist_ok=True)
        self.sigma = 1.0

        self.seg_metrics = SegMetrics(['ari', 'ari_fg'])
        self.recon_metrics = ReconMetrics(config.lpips_net)
        self.recon_rgb = config.get('recon_rgb', True)
        
       
    def configure_optimizers(self):
        warmup_steps = self.cfg.warmup_steps
        min_lr_factor = self.cfg.min_lr_factor
        decay_steps = self.cfg.decay_steps
        def lr_warmup_exp_decay(step: int):
            factor = min(1, step / (warmup_steps + 1e-6))
            decay_factor = 0.5 ** (step / decay_steps * 1.5)
            return factor * decay_factor * (1 - min_lr_factor) + min_lr_factor
        def lr_exp_decay(step: int):
            decay_factor = 0.5 ** (step / decay_steps)
            return decay_factor * (1 - min_lr_factor) + min_lr_factor
        params_nerf = [{'params': self.nerf.parameters(),
                        'lr': self.cfg.lr, 'weight_decay': self.cfg.weight_decay}]
        params_renderer = [{'params': self.renderer.parameters(),
                        'lr': self.cfg.lr, 'weight_decay': self.cfg.weight_decay}]
        params_slot_enc = [{'params': (x[1] for x in self.slot_3D.named_parameters() if 'dino' not in x[0]), 
                           'lr': self.cfg.lr, 'weight_decay': self.cfg.weight_decay}]
        params_slot_dec = [{'params': self.slot_dec.parameters(), 
                           'lr': self.cfg.lr, 'weight_decay': self.cfg.weight_decay}]
        params = params_nerf + params_renderer + params_slot_enc + params_slot_dec
        lr_lambda_list = [lr_exp_decay, lr_exp_decay, lr_warmup_exp_decay, lr_warmup_exp_decay]
        
        if self.cfg.grid_init == 'tensorf' or self.cfg.grid_init == '3d':
            params_grid = [{'params': [x[1] for x in self.renderer.named_parameters() if 'grid' in x[0]],
                            'lr': self.cfg.lr * 20, 'weight_decay': self.cfg.weight_decay}]
            params = params + params_grid
            lr_lambda_list = lr_lambda_list + [lr_exp_decay]
        
        if self.cfg.coarse_to_fine:
            params_nerf_fine = [{'params': self.nerf_fine.parameters(),
                                'lr': self.cfg.lr, 'weight_decay': self.cfg.weight_decay}]
            params_slot_dec_fine = [{'params': self.slot_dec_fine.parameters(),
                                    'lr': self.cfg.lr, 'weight_decay': self.cfg.weight_decay}]
            params = params + params_nerf_fine + params_slot_dec_fine
            lr_lambda_list = lr_lambda_list + [lr_exp_decay] + [lr_warmup_exp_decay]
        
        opt = Lion(params, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lr_lambda_list)

        return [opt], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, rays, depth_range, slots, view_feats, cam, src_rgbs, src_cams, is_train):
        B, Nr, _ = rays.shape
        outputs = []
        render_depth = not is_train
        render_ins = self.global_step >= self.instance_steps or not is_train
        render_color = self.recon_rgb or not is_train
        for i in range(0, Nr, self.cfg.chunk):
            outputs.append(self.renderer(self.nerf, self.nerf_fine, self.slot_dec, self.slot_dec_fine, 
                                         rays[:, i: i + self.cfg.chunk], depth_range, slots, 
                                         cam, src_rgbs, src_cams, view_feats,
                                         False, is_train, 
                                         render_color=render_color, render_depth=render_depth,
                                         render_ins=render_ins))
        keys = outputs[0].keys()
        out = {}
        for k in keys:
            if 'dist' in k or 'loss' in k:
                out[k] = torch.stack([o[k] for o in outputs], 0).mean()
            else:
                out[k] = torch.cat([o[k] for o in outputs], 1).flatten(0, 1)
        return out

    def training_step(self, batch, batch_idx):
        self.sigma = self.cosine_anneal(self.global_step, self.cfg.sigma_steps, final_value=0)
        if self.cfg.random_proj:
            ratio = self.cosine_anneal(self.global_step, self.cfg.random_proj_steps, start_value=0.99, final_value=0)
            self.nerf.random_proj_ratio = 1 - ratio
            self.log('ratio', 1-ratio)
        self.log('sigma', self.sigma)
        src_rgbs, src_cams = batch['src_rgbs'], batch['src_cams'] # [B, N, H, W, 3], [B, N, 34]
        B, N_views = src_rgbs.shape[:2]

        src_rgbs = src_rgbs.permute(0, 1, 4, 2, 3) # [B, Nv, 3, H, W]
        slots, attn, view_feats = self.slot_3D(sigma=self.sigma, images=src_rgbs, src_cams=src_cams) 

        rgbs = batch['rgbs'] # [B, Br, 3]
        rays = batch['rays'] # [B, Br, 6]
        depth_range = batch['depth_range'] # [B, 2]
        view_feats = view_feats.permute(0, 1, 4, 2, 3) # [B, Nv, D, H, W]
        output = self(rays, depth_range, slots, view_feats, None, src_rgbs, src_cams, True)

        loss_rgb = self.loss(output['rgb_c'], rgbs.view(-1, 3))
        if self.cfg.coarse_to_fine:
            loss_rgb = (loss_rgb + self.loss(output['rgb_f'], rgbs.view(-1, 3))) / 2
        if self.cfg.normalize:
            loss_rgb = loss_rgb / 4
        loss = loss_rgb
        self.log("train/loss_rgb", loss_rgb, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        self.recon_metrics.set_divice(self.device)

    def validation_step(self, batch, batch_idx):
        out_put = {}

        src_rgbs, src_cams = batch['src_rgbs'], batch['src_cams'] # [B, N, H, W, 3], [B, N, 34]
        
        B, N_views, H, W, _ = src_rgbs.shape
        src_rgbs = src_rgbs.permute(0, 1, 4, 2, 3) # [B, Nv, 3, H, W]
        slots, attn, view_feats = self.slot_3D(sigma=0, images=src_rgbs, src_cams=src_cams) 

        # get rays from cameras
        N = batch['cam'].shape[1]
        rays = batch['rays'].view(1, -1, 6) # [1, Br, 6]
        depth_range = batch['depth_range'] # [B, 2]
        view_feats = view_feats.permute(0, 1, 4, 2, 3) # [B, Nv, D, H, W]
        output = self(rays, depth_range, slots, view_feats, None, src_rgbs, src_cams, False)

        if self.cfg.coarse_to_fine:
            output_rgb = output['rgb_f']
            output_instances = output['instance_f']
        else:
            output_rgb = output['rgb_c']
            output_instances = output['instance_c']

        shape = (N, H, W, 3)
        rgbs = batch['rgbs'].view(N, H*W, -1) # [1, N*H*W, 3]
        if self.cfg.normalize:
            output_rgb = output_rgb * 0.5 + 0.5
            rgbs = rgbs * 0.5 + 0.5
        if self.cfg.dataset == 'dtu' or self.cfg.dataset == 'ibrnet':
            recon_metrics = self.recon_metrics(output_rgb.view(shape).permute(0, 3, 1, 2), rgbs.view(shape).permute(0, 3, 1, 2))
            out_put.update(recon_metrics)
        else:
            rs_instances = batch['instances'].view(N, -1) # [N, H*W]
            if self.cfg.dataset != 'scannet' and self.cfg.dataset != 'oct':
                Nv = self.cfg.num_src_view
                recon_metrics = self.recon_metrics(output_rgb.view(shape)[Nv:].permute(0, 3, 1, 2), rgbs.view(shape)[Nv:].permute(0, 3, 1, 2))
                seg_metrics = self.seg_metrics(output_instances.view(N, H*W, -1)[:Nv], rs_instances[:Nv])
                out_put.update(recon_metrics)
                out_put.update(seg_metrics)
                # src_metircs = self.recon_metrics(output_rgb.view(shape)[:Nv].permute(0, 3, 1, 2), rgbs.view(shape)[:Nv].permute(0, 3, 1, 2))
                # for key, value in src_metircs.items():
                #     out_put['src_' + key] = value
                nv_seg_metrics = self.seg_metrics(output_instances.view(N, H*W, -1)[Nv:], rs_instances[Nv:])
                for key, value in nv_seg_metrics.items():
                    out_put['nv_' + key] = value
            else:
                recon_metrics = self.recon_metrics(output_rgb.view(shape).permute(0, 3, 1, 2), rgbs.view(shape).permute(0, 3, 1, 2))
                K = output_instances.shape[-1]
                seg_metrics = self.seg_metrics(output_instances.reshape(N, -1, K), rs_instances)
                out_put.update(recon_metrics)
                out_put.update(seg_metrics)
        
        return out_put

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        logs = {}
        for k in keys:
            v = torch.stack([x[k] for x in outputs]).mean()
            logs['val/' + k] = v
        self.log_dict(logs, sync_dist=True)
        table = [keys, ]
        table.append(tuple([logs['val/' + key] for key in table[0]]))
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        self.visualize(self.val_dataloader())
        
    @rank_zero_only
    def visualize(self, dataloader):
        (self.output_dir_result_seg / f"{self.global_step:06d}").mkdir(exist_ok=True)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx in self.cfg.visualized_indices:
                cam = batch['cam'].reshape(1, -1, 34).to(self.device) # [1, N, 34]
                rays = batch['rays'].reshape(1, -1, 6).to(self.device) # [1, NHW, 6]
                NHW = rays.shape[1]
                instances = batch.get('instances', torch.zeros(NHW))
                rgbs = batch.get('rgbs', torch.zeros([NHW, 3]))
                depth = batch.get('depth', torch.zeros_like(instances))
                semantics = batch.get('semantics', torch.zeros_like(instances))
                src_rgbs, src_cams = batch['src_rgbs'].to(self.device), batch['src_cams'].to(self.device)
                B, N_views, H, W, _ = src_rgbs.shape
                src_rgbs = src_rgbs.permute(0, 1, 4, 2, 3) # [B, Nv, 3, H, W]
                slots, attn, view_feats = self.slot_3D(sigma=self.sigma, images=src_rgbs, src_cams=src_cams) 
                view_feats = view_feats.permute(0, 1, 4, 2, 3) # [B, Nv, D, H, W]
                depth_range = batch['depth_range'].to(self.device) # [B, 2]
                output = self(rays, depth_range, slots, view_feats, None, src_rgbs, src_cams, False)
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
                    rgbs = rgbs * 0.5 + 0.5
                    src_rgbs = src_rgbs * 0.5 + 0.5
                images = src_rgbs[0].reshape(-1, 3, H, W).cpu() # [N, 3, H, W]
                N = cam.shape[1]
                Nv = self.cfg.num_src_view
                n_pad = 4 - Nv % 4 if Nv % 4 != 0 else 0
                for frame_id in range(N):
                    stack = visualize_panoptic_outputs(output_rgb.view(N, H*W, -1)[frame_id], output_instances.view(N, H*W, -1)[frame_id], 
                                                       output_depth.view(N, H*W)[frame_id], rgbs.view(N, H*W, -1)[frame_id], 
                                                       semantics.view(N, H*W)[frame_id], instances.view(N, H*W)[frame_id], 
                                                       H, W, depth.view(N, H*W)[frame_id])
                    stack = torch.cat([images, torch.zeros(n_pad, 3, H, W), stack], dim=0)
                    if self.cfg.logger == 'wandb':
                        self.logger.log_image(key=f"images/{batch_idx:04d}_{frame_id:04d}", images=[make_grid(stack, value_range=(0, 1), nrow=4, normalize=True)])
                    save_image(stack, self.output_dir_result_images / f"{self.global_step:06d}_{batch_idx:04d}_{frame_id:04d}.jpg", value_range=(0, 1), nrow=4, normalize=True)
                if self.cfg.dataset == 'dtu':
                    H1, W1 = 76, 100
                else:
                    H1, W1 = H // 4, W // 4
                attn = attn.reshape(-1, src_rgbs.shape[1], H1, W1)
                attn = F.interpolate(attn, size=(H, W), mode='nearest')
                attn = attn.permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 3, 1, 1)
                K = attn.shape[1]
                img = torch.cat([images.unsqueeze(1).cpu(), 1 - attn.cpu()], dim=1).reshape(-1, 3, H, W)
                if self.cfg.logger == 'wandb':
                    self.logger.log_image(key=f"attn/{batch_idx:04d}", images=[make_grid(img, value_range=(0, 1), nrow=K+1, normalize=True)])

    def on_test_epoch_start(self):
        torch.cuda.empty_cache()
        self.recon_metrics.set_divice(self.device)

    def test_step(self, batch, batch_idx):
        out_put = {}

        src_rgbs, src_cams = batch['src_rgbs'], batch['src_cams'] # [B, N, H, W, 3], [B, N, 34]
        B, Nv, H, W, _ = src_rgbs.shape
        src_rgbs = src_rgbs.permute(0, 1, 4, 2, 3) # [B, Nv, 3, H, W]
        slots, attn, view_feats = self.slot_3D( sigma=0, images=src_rgbs, src_cams=src_cams) 

        images = src_rgbs[0].reshape(-1, 3, H, W).cpu() # [N_src_view, 3, H, W]
        H1, W1 = H // 4, W // 4
        if self.cfg.dataset == 'dtu':
            H1 = 76
        attn = attn.reshape(-1, src_rgbs.shape[1], H1, W1)
        attn = F.interpolate(attn, size=(H, W), mode='nearest')
        attn = attn.permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 3, 1, 1)
        img = torch.cat([images.unsqueeze(1).cpu(), 1 - attn.cpu()], dim=1).reshape(-1, 3, H, W)
        img = make_grid(img, nrow=Nv+1, padding=0) # [3, (N_src_view+1)*H, W]
        save_img_from_tensor(img.permute(1, 2, 0), self.output_dir_result_attn / f"{batch_idx:04d}_attn.png")

        # get rays from cameras
        cam = batch['cam'] # [1, N, 34]
        N = cam.shape[1]
        rays = batch['rays'].view(1, -1, 6) # [1, Br, 6]
        depth_range = batch['depth_range'] # [B, 2]

        view_feats = view_feats.permute(0, 1, 4, 2, 3) # [B, Nv, D, H, W]
        output = self(rays, depth_range, slots, view_feats, None, src_rgbs, src_cams, False)
        if self.cfg.coarse_to_fine:
            output_rgb = output['rgb_f']
            output_instances = output['instance_f']
            output_depth = output['depth_f']
        else:
            output_rgb = output['rgb_c']
            output_instances = output['instance_c']
            output_depth = output['depth_c']

        shape = (N, H, W, 3)
        rgbs = batch['rgbs'].view(N, H*W, -1) # [1, N*H*W, 3]
        if self.cfg.normalize:
            output_rgb = output_rgb * 0.5 + 0.5
            rgbs = rgbs * 0.5 + 0.5
        if self.cfg.dataset == 'dtu':
            recon_metrics = self.recon_metrics(output_rgb.view(shape).permute(0, 3, 1, 2), rgbs.view(shape).permute(0, 3, 1, 2))
            out_put.update(recon_metrics)
        else:
            rs_instances = batch['instances'].view(N, -1) # [N, H*W]
            if self.cfg.dataset != 'scannet':
                Nv = self.cfg.num_src_view
                recon_metrics = self.recon_metrics(output_rgb.view(shape)[Nv:].permute(0, 3, 1, 2), rgbs.view(shape)[Nv:].permute(0, 3, 1, 2))
                seg_metrics = self.seg_metrics(output_instances.view(N, H*W, -1)[:Nv], rs_instances[:Nv])
                out_put.update(recon_metrics)
                out_put.update(seg_metrics)
                src_metircs = self.recon_metrics(output_rgb.view(shape)[:Nv].permute(0, 3, 1, 2), rgbs.view(shape)[:Nv].permute(0, 3, 1, 2))
                for key, value in src_metircs.items():
                    out_put['src_' + key] = value
                nv_seg_metrics = self.seg_metrics(output_instances.view(N, H*W, -1)[Nv:], rs_instances[Nv:])
                for key, value in nv_seg_metrics.items():
                    out_put['nv_' + key] = value
            else:
                recon_metrics = self.recon_metrics(output_rgb.view(shape).permute(0, 3, 1, 2), rgbs.view(shape).permute(0, 3, 1, 2))
                K = output_instances.shape[-1]
                seg_metrics = self.seg_metrics(output_instances.reshape(N, -1, K), rs_instances)
                out_put.update(recon_metrics)
                out_put.update(seg_metrics)
        
        print(f'batch_idx: {batch_idx}')
        for k, v in recon_metrics.items():
            print(k, ': ', v.item())
        for k, v in seg_metrics.items():
            print(k, ': ', v.item())
        print('-' * 40)
        # save img
        imgs_gt = rgbs.view(shape)
        imgs_pred = output_rgb.view(shape)
        # seg_gt = rs_instances.view(N, H, W)
        seg_pred = output_instances.argmax(-1).view(N, H, W)
        depth_pred = output_depth.view(N, H, W)
        for n in range(N):
            save_img_from_tensor(imgs_gt[n], self.output_dir_result_images / f"{batch_idx:04d}_{n:02d}_rgb_gt.png")
            save_img_from_tensor(imgs_pred[n], self.output_dir_result_images / f"{batch_idx:04d}_{n:02d}_rgb_pred.png")
            # save_seg_from_tensor(seg_gt[n], self.output_dir_result_seg / f"{batch_idx:04d}_{n:02d}_seg_gt.png")
            save_seg_from_tensor(seg_pred[n], self.output_dir_result_seg / f"{batch_idx:04d}_{n:02d}_seg_pred.png")
            depth = visualize_depth(depth_pred[n], use_global_norm=False) # [3, H, W]
            save_img_from_tensor(depth.permute(1, 2, 0), self.output_dir_result_depth / f"{batch_idx:04d}_{n:02d}_depth_pred.png")
        return out_put

    def test_epoch_end(self, outputs):
        keys = outputs[0].keys()
        logs = {}
        for k in keys:
            v = torch.stack([x[k] for x in outputs]).mean()
            logs['test/' + k] = v
        self.log_dict(logs, sync_dist=True)
        table = [keys, ]
        table.append(tuple([logs['test/' + key] for key in table[0]]))
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        self.visualize(self.test_dataloader())

    def train_dataloader(self):
        shuffle = False if self.cfg.job_type == 'debug' else True
        shuffle = shuffle and self.train_sampler is None
        persistent_workers = self.cfg.num_workers > 0
        return DataLoader(self.train_set, self.cfg.batch_size, shuffle=shuffle, pin_memory=True, num_workers=self.cfg.num_workers, sampler=self.train_sampler, persistent_workers=persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()
        self.slot_dec.force_bg = self.cfg.force_bg and self.global_step < self.cfg.force_bg_steps
        if self.cfg.coarse_to_fine:
            print(f'Using {self.renderer.n_samples} points for coarse rendering, {self.renderer.n_samples_fine} points for fine rendering')
        else:
            print(f'Using {self.renderer.n_samples} points for rendering')
    
    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
        if start_value <= final_value or start_step >= final_step:
            return final_value
        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value


@hydra.main(config_path='../config', config_name='config', version_base='1.2')
def main(config):
    config = config.cfg
    trainer = create_trainer(config)
    model = TensoRFTrainer(config)
    if trainer.logger is not None and config.watch_model:
        trainer.logger.watch(model)
    if config.job_type == 'test':
        ckpt = torch.load(config.ckpt_path) 
        model.load_state_dict(ckpt['state_dict'])
        print(f'Load from checkpoint: {config.ckpt_path}')
        trainer.test(model)
    elif config.job_type == 'vis':
        ckpt = torch.load(config.ckpt_path) 
        model.load_state_dict(ckpt['state_dict'])
        print(f'Load from checkpoint: {config.ckpt_path}')
        model.eval()
        with torch.no_grad():
            model.visualize(model.val_dataloader())
    else:
        trainer.fit(model)
        trainer.test(model)


if __name__ == '__main__':
    main()
