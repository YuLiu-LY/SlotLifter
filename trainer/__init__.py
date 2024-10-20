# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
from pathlib import Path
from random import randint
import datetime

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from util.distinct_colors import DistinctColors
from util.misc import visualize_depth, get_boundary_mask
from util.filesystem_logger import FilesystemLogger
import torch.nn.functional as F


def generate_experiment_name(config):
    if config.resume is not None and config.job_type == 'train':
        experiment = Path(config.resume).parents[1].name
        # experiment = f"{config.exp_name}_{datetime.datetime.now().strftime('%m%d%H%M')}"
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{config.exp_name}_{datetime.datetime.now().strftime('%m%d%H%M')}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


def create_trainer(config):

    config.exp_name = generate_experiment_name(config)
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed, workers=True)

    # save code files
    filesystem_logger = FilesystemLogger(config)

    if config.logger == 'wandb':
        logger = WandbLogger(
            project=config.project,
            entity=config.entity,
            group=config.group,
            name=config.exp_name,
            job_type=config.job_type,
            tags=config.tags,
            notes=config.notes,
            id=config.exp_name, 
            # settings=wandb.Settings(start_method='thread'),
        )
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(dirpath=(Path(config.log_path) / config.exp_name / "checkpoints"),
                                          monitor=f'val/{config.monitor}',
                                          save_top_k=1,
                                          save_last=True,
                                          mode='max',
                                          )
    callbacks = [LearningRateMonitor("step"),  checkpoint_callback] if logger else []
    gpu_count = torch.cuda.device_count()
    if config.job_type == 'debug':
        config.train_percent = 30
        config.val_percent = 1
        config.test_percent = 1
        config.val_check_interval = 1

    kwargs = {
        'resume_from_checkpoint': config.resume,
        'logger': logger,
        'accelerator': 'gpu',
        'devices': gpu_count,
        'strategy': 'ddp' if gpu_count > 1 else None,
        'num_sanity_val_steps': 1,
        'max_steps': config.max_steps,
        'max_epochs': config.max_epochs,
        'limit_train_batches': config.train_percent,
        'limit_val_batches': config.val_percent,
        'limit_test_batches': config.test_percent,
        'val_check_interval': float(min(config.val_check_interval, 1)),
        'check_val_every_n_epoch': max(1, config.val_check_interval),
        'callbacks': callbacks,
        'gradient_clip_val': config.grad_clip if config.grad_clip > 0 else None,
        'precision': config.precision,
        'profiler': config.profiler,
        'benchmark': config.benchmark,
        'deterministic': config.deterministic,
    }
    trainer = Trainer(**kwargs)
    return trainer


def visualize_panoptic_outputs(p_rgb, p_instances, p_depth, rgb, semantics, instances, H, W, depth=None, p_semantics=None):
    alpha = 0.65
    distinct_colors = DistinctColors()
    p_rgb = p_rgb.cpu()
    img = p_rgb.view(H, W, 3).cpu().permute(2, 0, 1)

    depth_scale = 1
    if p_depth is not None:
        p_depth = visualize_depth(p_depth.view(H, W) * depth_scale, use_global_norm=False)
    else:
        p_depth = torch.zeros_like(img)
    if depth is not None:
        depth = visualize_depth(depth.view(H, W) * depth_scale, use_global_norm=False)
    else:
        depth = torch.zeros_like(img)

    def get_color(p_instances, idx_bg, im, rgb_):
        colored_img_instance = distinct_colors.apply_colors_fast_torch(p_instances).float()
        # boundaries_img_instances = get_boundary_mask(p_instances.view(H, W))
        # colored_img_instance[p_instances == idx_bg, :] = rgb_[p_instances == idx_bg, :]
        img_instances = colored_img_instance.view(H, W, 3).permute(2, 0, 1) * alpha + im * (1 - alpha)
        # img_instances[:, boundaries_img_instances > 0] = 0
        return img_instances

    img_gt = rgb.view(H, W, 3).permute(2, 0, 1)
    idx_bg = p_instances.sum(0).argmax().item()
    p_instances = p_instances.argmax(dim=1).cpu()
    img_instances = get_color(p_instances, idx_bg, img, p_rgb)

    if semantics is not None and semantics.max() > 0:
        img_semantics_gt = distinct_colors.apply_colors_fast_torch(semantics).view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
        boundaries_img_semantics_gt = get_boundary_mask(semantics.view(H, W))
        img_semantics_gt[:, boundaries_img_semantics_gt > 0] = 0
    else:
        img_semantics_gt = torch.zeros_like(img_gt)
    if p_semantics is not None and p_semantics.max() > 0:
        p_semantics = p_semantics.argmax(dim=1).cpu()
        img_semantics = distinct_colors.apply_colors_fast_torch(p_semantics).view(H, W, 3).permute(2, 0, 1) * alpha + img * (1 - alpha)
        boundaries_img_semantics = get_boundary_mask(p_semantics.view(H, W))
        img_semantics[:, boundaries_img_semantics > 0] = 0
    else:
        img_semantics = torch.zeros_like(img_gt)
    if instances is not None and instances.max() > 0:
        img_instances_gt = get_color(instances.long(), 0, img_gt, rgb)
    else:
        img_instances_gt = torch.zeros_like(img_gt)
    stack = torch.cat([torch.stack([img_gt, img_semantics_gt, img_instances_gt, depth]), torch.stack([img, img_semantics, img_instances, p_depth])], dim=0)
    return stack