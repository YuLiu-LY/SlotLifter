# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import hydra
import json
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
from torch.utils.data import Dataset
from util.camera import compute_world2normscene
from util.misc import SubSampler
from util.ray import get_rays
from dataset.data_utils import get_nearest_pose_ids


def move_left_zero(x):
    return '0' if int(x) == 0 else x.lstrip('0')


class ScannetDataset(Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.split = 'test' if split != 'train' else 'train'
        self.root_dir = Path(cfg.dataset_root)
        if self.split == 'test':
            self.root_dir = Path(cfg.dataset_root.replace('scannet', 'scannet_test'))
        self.img_size = cfg.img_size[0], cfg.img_size[1]

        self.img_root = f'color_480640'

        self.train_subsample_frames = cfg.train_subsample_frames
        self.num_src_view = cfg.num_src_view
        self.world2scene = np.eye(4, dtype=np.float32)
        self.depth_range = [0.1, 10]
        self.max_depth = 10
        self.normalize = cfg.normalize
        self.ray_batchsize = cfg.ray_batchsize
        self.norm_scene = cfg.norm_scene
        self.subsampler = SubSampler()
        self.setup_data()
    
    def __len__(self):
        return sum([len(x) for x in self.indices])

    def setup_data(self):
        scenes = sorted(glob(f'{str(self.root_dir)}/*'))
        test_scenes = ["scene0000_01",
                       "scene0079_00",
                       "scene0158_00", 
                       "scene0316_00",
                       "scene0521_00",
                       "scene0553_00",
                       "scene0616_00", 
                       "scene0653_00",
                       ]
        self.scenes = scenes
        print(f"Using {len(self.scenes)} scenes for {self.split}")
        scene_names = [x.split('/')[-1] for x in self.scenes]
        print(self.split, scene_names)
        self.train_indices = []
        self.val_indices = []
        self.all_frame_names = [] 
        self.cam2normscene = []
        self.cam2scenes = []
        self.intrinsics = []
        self.scene2normscene = []
        self.normscene_scale = []
        self.segmentation_data = []
        
        for idx, scene_dir in enumerate(self.scenes):
            self.setup_data_one_scene(idx)
        if self.split == 'train':
            self.indices = self.train_indices
        else:
            self.indices = self.val_indices
        self.frame_idx2scene_idx = {}
        self.frame_idx2sample_idx = {}
        frame_idx = 0
        for scene_idx, scene_indices in enumerate(self.indices):
            for _, sample_idx in enumerate(scene_indices):
                self.frame_idx2scene_idx[frame_idx] = scene_idx
                self.frame_idx2sample_idx[frame_idx] = sample_idx
                frame_idx += 1
        if self.norm_scene:
            self.depth_range = self.depth_range[0] * min(self.normscene_scale), self.depth_range[1] * max(self.normscene_scale)
        print(f"Depth range: {self.depth_range}")

    def setup_data_one_scene(self, scene_idx):
        scene_dir = self.scenes[scene_idx]
        # split
        if self.split == 'train' or self.split == 'val':
            frames = json.loads((Path(scene_dir) / "sampled_frames.json").read_text())
            scene_frames = sorted([x.zfill(4) for x in frames])
            scene_frames = [move_left_zero(x) for x in scene_frames]
            sample_indices = list(range(len(scene_frames)))
            val_indices = sample_indices[::8]
            train_indices = [x for x in sample_indices if x not in val_indices]
        else: 
            train_frames = np.loadtxt(Path(scene_dir) / "train.txt", dtype=str)
            val_frames = np.loadtxt(Path(scene_dir) / "test.txt", dtype=str)
            scene_frames = train_frames.tolist() + val_frames.tolist()
            scene_frames = [x.split('.')[0] for x in scene_frames]
            train_indices = list(range(len(train_frames)))
            val_indices = list(range(len(train_frames), len(scene_frames)))
            sample_indices = list(range(len(scene_frames)))

        # print(f"Loading {scene_name}")
        self.train_indices.append(train_indices)
        self.val_indices.append(val_indices)
        self.all_frame_names.append(scene_frames)
        
        dims, cam2scene = [], []
        img_h, img_w = 968, 1296
        intrinsic_color = np.array([[float(y.strip()) for y in x.strip().split()] for x in (Path(scene_dir) / f"intrinsic" / "intrinsic_color.txt").read_text().splitlines() if x != ''])
        intrinsic_color = torch.from_numpy(intrinsic_color[:3, :3]).float()
        scale_x, scale_y = self.img_size[1] / img_w, self.img_size[0] / img_h
        intrinsic_normed = torch.diag(torch.Tensor([scale_x, scale_y, 1])) @ intrinsic_color
        intrinsic = torch.eye(4)
        intrinsic[:3, :3] = intrinsic_normed
        self.intrinsics.append(intrinsic)
        
        for sample_index in sample_indices:
            cam2world = np.array([[float(y.strip()) for y in x.strip().split()] for x in (Path(scene_dir) / f"pose" / f"{scene_frames[sample_index]}.txt").read_text().splitlines() if x != ''])
            cam2world = torch.from_numpy(self.world2scene @ cam2world).float()
            cam2scene.append(cam2world)
            dims.append([img_h, img_w])

        cam2scene = torch.stack(cam2scene)
        intrinsics = intrinsic_color.unsqueeze(0).expand(len(cam2scene), -1, -1)
        scene2normscene = compute_world2normscene(
            torch.Tensor(dims).float(),
            intrinsics,
            cam2scene,
            max_depth=self.max_depth,
            rescale_factor=1.0
        )
        self.scene2normscene.append(scene2normscene)
        self.normscene_scale.append(scene2normscene[0, 0])
        cam2normscene = []
        for sample_index in sample_indices:
            cam2normscene.append(scene2normscene @ cam2scene[sample_index])
        cam2normscene = torch.stack(cam2normscene)
        
        self.cam2normscene.append(cam2normscene)
        self.cam2scenes.append(cam2scene)

    def load_sample(self, sample_index, scene_idx):
        scene_dir = self.scenes[scene_idx]
        image = Image.open(Path(scene_dir) / f"{self.img_root}" / f"{self.all_frame_names[scene_idx][sample_index]}.jpg")
        # image = image.resize(self.img_size[::-1], Image.BILINEAR)
        image = torch.from_numpy(np.array(image) / 255).float() # [H, W, 3]
        if self.normalize:
            image = image * 2 - 1 # [-1, 1]
        return image.view(-1, 3)
    
    def sample_support_views(self, pose, scene_idx):
        # sample support ids
        support_indices = self.train_indices[scene_idx]
        if self.split == 'train':
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            nearest_pose_ids = get_nearest_pose_ids(pose.numpy(),
                                                    self.cam2scenes[scene_idx][support_indices].numpy(),
                                                    min(self.num_src_view * subsample_factor, 22),
                                                    tar_id=0,
                                                    angular_dist_method='mix')
            nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_src_view, replace=False)
        else:
            nearest_pose_ids = get_nearest_pose_ids(pose.numpy(),
                                                    self.cam2scenes[scene_idx][support_indices].numpy(),
                                                    self.num_src_view,
                                                    tar_id=-1,
                                                    angular_dist_method='mix')
        nearest_pose_ids = torch.from_numpy(nearest_pose_ids).long()
        
        return nearest_pose_ids

    def __getitem__(self, idx):
        scene_idx = self.frame_idx2scene_idx[idx]
        sample_idx = self.frame_idx2sample_idx[idx]
        sample = {}

        # sample support ids
        poses = self.cam2normscene[scene_idx] if self.norm_scene else self.cam2scenes[scene_idx]
        pose = self.cam2scenes[scene_idx][sample_idx]
        nearest_pose_ids = self.sample_support_views(pose, scene_idx)
        support_indices = self.train_indices[scene_idx]
        src_rgbs, src_cams = [], []
        for i in nearest_pose_ids:
            id = support_indices[i]
            src_cam = torch.cat([torch.Tensor(self.img_size), self.intrinsics[scene_idx].flatten(),
                                    poses[id].flatten()])
            src_rgb = self.load_sample(id, scene_idx)
            src_rgbs.append(src_rgb)
            src_cams.append(src_cam)
        src_rgbs = torch.stack(src_rgbs) # [N, H*W, 3]
        src_cams = torch.stack(src_cams) # [N, 34]

        rgbs = self.load_sample(sample_idx, scene_idx)   
        cam = torch.cat([torch.Tensor(self.img_size), self.intrinsics[scene_idx].flatten(),
                                    poses[sample_idx].flatten()])
        tgt_cam = cam[None] # [1, 34]
        tgt_rays = get_rays(tgt_cam, *self.img_size).view(-1, 6) # [HW, 3]
        H, W = self.img_size
        N = self.num_src_view
        if self.split == 'train':
            # subsample
            Br = self.ray_batchsize
            subsample_idx = self.subsampler.idx_subsample(self.img_size, Br)
            sample['idx'] = subsample_idx # [Br, 1]
            tgt_rgbs = rgbs.gather(0, subsample_idx.expand(-1, 3)) # [Br, 3]
            tgt_rays = tgt_rays.gather(0, subsample_idx.expand(-1, 6)) # [Br, 3]
        else:
            tgt_rgbs = rgbs[None]# [1, HW, 3]
            tgt_rays = tgt_rays[None] # [1, HW, 3]
            semantics = Image.open(Path(self.scenes[scene_idx]) / "semantics" / f"{self.all_frame_names[scene_idx][sample_idx]}.png")
            semantics = torch.tensor(np.array(semantics.resize(self.img_size[::-1], Image.Resampling.NEAREST), np.int32)).long().reshape(-1)
            instances = Image.open(Path(self.scenes[scene_idx]) / "instance" / f"{self.all_frame_names[scene_idx][sample_idx]}.png")
            instances = torch.tensor(np.array(instances.resize(self.img_size[::-1], Image.Resampling.NEAREST), np.int32)).long().reshape(-1)
            sample['semantics'] = semantics # [H*W]
            sample['instances'] = instances # [H*W]
        sample['rgbs'] = tgt_rgbs # [Br, 3] or [N1, HW, 3]
        sample['rays'] = tgt_rays # [Br, 3] or [N1, HW, 3]
        sample['cam'] = tgt_cam # [Br, 34]  or [N1, 34]
        sample['src_rgbs'] = src_rgbs.reshape(N, H, W, 3) # [N, HW，3] 
        sample['src_cams'] = src_cams # [N, 34] 2+16+16
        sample['depth_range'] = torch.tensor(self.depth_range).float()
        return sample



## test
from torch.utils.data import DataLoader

@hydra.main(config_path='../config/cfg', config_name='scannet', version_base='1.2')
def main(config):
    config.img_size = [480, 640]
    config.num_workers = 0
    train_set = ScannetDataset(config, "train")
    test_set = ScannetDataset(config, "test")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config.num_workers)
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        B = config.batch_size
        print(batch['rgbs'].shape)
        break
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        print(batch['instances'].unique())
        break

# test
if __name__ == '__main__':
    main()