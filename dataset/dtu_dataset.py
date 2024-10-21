import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from pathlib import Path
import torch
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from util.misc import SubSampler
from pathlib import Path
from util.ray import get_rays
from dataset.data_utils import get_nearest_pose_ids


class DTUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        split="train",
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param split train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        """
        super().__init__()
        self.data_root = cfg.dataset_root
        self.normalize = cfg.normalize
        self.depth_range = [0.1, 5.0]
        self.split = split
        self.img_root = f'image'
        self.ray_batchsize = cfg.ray_batchsize
        self.subsampler = SubSampler()
        self.num_src_view = cfg.num_src_view

        self.scene_id = cfg.scene_id

        self.img_size = cfg.img_size[0], cfg.img_size[1]
        # sub_format == "dtu":
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

        self.setup_data()
    
    def __len__(self):
        return sum([len(x) for x in self.indices])

    def setup_data(self):
        path =  "new_val.lst" if self.split != "train" else "new_train.lst"
        scene_list_path = os.path.join(self.data_root, path)
        scenes = []
        with open(scene_list_path, "r") as f:
            for line in f:
                scenes.append(line.strip())

        self.scenes = [os.path.join(self.data_root, scene) for scene in scenes]
        if self.scene_id != -1:
            self.scenes = [self.scenes[self.scene_id]]
        self.train_indices = []
        self.val_indices = []
        self.all_frame_names = [] 
        self.cam2normscene = []
        self.cam2scenes = []
        self.intrinsics = []
        self.scene2normscene = []
        self.normscene_scale = []
        self.all_cams = []
        for scene in tqdm(self.scenes, desc="Loading DTU dataset"):
            self.setup_one_scene(scene)
        print(
            "Loading DTU dataset from", self.data_root,
            'found', len(self.scenes),
            self.split,"scenes",
        )
        if self.split == 'train':
            self.indices = self.train_indices
        else:
            self.indices = self.val_indices
        # self.indices = [list(range(len(self.all_frame_names[0])))]
        self.frame_idx2scene_idx = {}
        self.frame_idx2sample_idx = {}
        frame_idx = 0
        for scene_idx, scene_indices in enumerate(self.indices):
            for _, sample_idx in enumerate(scene_indices):
                self.frame_idx2scene_idx[frame_idx] = scene_idx
                self.frame_idx2sample_idx[frame_idx] = sample_idx
                frame_idx += 1
        print("depth range", self.depth_range)

    def setup_one_scene(self, scene):
        all_frames = sorted(glob(os.path.join(scene, self.img_root, "*")))
        all_frames = [x.split("/")[-1].split(".")[0] for x in all_frames]
        self.all_frame_names.append(all_frames)

        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
        cam2scene = []
        dims = []
        img_w, img_h = Image.open(os.path.join(scene, self.img_root, f'{all_frames[0]}.png')).size
        for i in range(len(all_frames)):
            cams = np.load(os.path.join(self.data_root, scene, "cameras.npz"))

            # Decompose projection matrix
            P = cams["world_mat_" + str(i)]
            P = P[:3]
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K = K / K[2, 2]

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]

            scale_mtx = cams.get("scale_mat_" + str(i))
            if scale_mtx is not None:
                norm_trans = scale_mtx[:3, 3:]
                norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]
                pose[:3, 3:] -= norm_trans
                pose[:3, 3:] /= norm_scale

            fx += torch.tensor(K[0, 0])
            fy += torch.tensor(K[1, 1])
            cx += torch.tensor(K[0, 2])
            cy += torch.tensor(K[1, 2])

            pose = torch.tensor(pose, dtype=torch.float32)
            cam2scene.append(pose)
            dims.append([img_h, img_w])
        fx /= len(all_frames)
        fy /= len(all_frames)
        cx /= len(all_frames)
        cy /= len(all_frames)
        intrinsic = torch.tensor([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        cam2scene = torch.stack(cam2scene).float()
        intrinsics = intrinsic.unsqueeze(0).expand(len(cam2scene), -1, -1).float()
        cams = torch.cat([
            torch.Tensor(self.img_size).expand(len(cam2scene), -1),
            intrinsics.flatten(1, 2),
            cam2scene.flatten(1, 2)], dim=1)
        self.all_cams.append(cams)
        self.cam2scenes.append(cam2scene)

        if self.split == "train":
            indices = list(range(len(all_frames)))
            self.train_indices.append(indices)
        else:
            indices = list(range(len(all_frames)))
            val_indices = indices[::8]
            train_indices = [i for i in indices if i not in val_indices]
            self.val_indices.append(val_indices)
            self.train_indices.append(train_indices)

    def load_sample(self, sample_index, scene_idx):
        scene_dir = self.scenes[scene_idx]
        image = Image.open(Path(scene_dir) / f"{self.img_root}" / f"{self.all_frame_names[scene_idx][sample_index]}.png")
        image = torch.from_numpy(np.array(image) / 255).float() # [H, W, 3]
        if self.normalize:
            image = image * 2 - 1 # [-1, 1]
        return image.view(-1, 3)
    
    def sample_support_views(self, pose, scene_idx):
        # sample support ids
        support_indices = self.train_indices[scene_idx]
        if self.split == 'train':
            subsample_factor = 1
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
        pose = self.cam2scenes[scene_idx][sample_idx]
        nearest_pose_ids = self.sample_support_views(pose, scene_idx)
        support_indices = self.train_indices[scene_idx]
        src_rgbs, src_cams = [], []
        for i in nearest_pose_ids:
            id = support_indices[i]
            src_cam = self.all_cams[scene_idx][id]
            src_rgb = self.load_sample(id, scene_idx)
            src_rgbs.append(src_rgb)
            src_cams.append(src_cam)
        src_rgbs = torch.stack(src_rgbs) # [N, H*W, 3]
        src_cams = torch.stack(src_cams) # [N, 34]

        rgbs = self.load_sample(sample_idx, scene_idx)   
        tgt_cam = self.all_cams[scene_idx][[sample_idx]]
        tgt_rays = get_rays(tgt_cam, *self.img_size).view(-1, 6) # [HW, 3]
        H, W = self.img_size
        N = self.num_src_view
        if self.split == 'train':
            # subsample
            Br = self.ray_batchsize
            subsample_idx = self.subsampler.idx_subsample(self.img_size, Br)
            tgt_rgbs = rgbs.gather(0, subsample_idx.expand(-1, 3)) # [Br, 3]
            tgt_rays = tgt_rays.gather(0, subsample_idx.expand(-1, 6)) # [Br, 3]
        else:
            tgt_rgbs = rgbs[None] # [1, HW, 3], [1, HW]
            tgt_rays = tgt_rays[None] # [1, HW, 3]
        sample['rgbs'] = tgt_rgbs # [Br, 3] or [N1, HW, 3]
        sample['rays'] = tgt_rays # [Br, 3] or [N1, HW, 3]
        sample['cam'] = tgt_cam # [Br, 34]  or [N1, 34]
        sample['src_rgbs'] = src_rgbs.reshape(N, H, W, 3) # [N, HW，3] 
        sample['src_cams'] = src_cams # [N, 34] 2+9+16
        sample['depth_range'] = torch.tensor(self.depth_range).float()
        return sample


import hydra
from torch.utils.data import DataLoader

@hydra.main(config_path='../config/cfg', config_name='dtu', version_base='1.2')
def main(config):
    config.num_workers = 0
    config.lambda_depth = 0.1

    train_set = DTUDataset(config, "train")
    val_set = DTUDataset(config, "val")
    test_set = DTUDataset(config, "test")

    train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config.num_workers)
    for i, batch in tqdm(enumerate(train_loader), total=len(train_set)):
        print(batch['cam'].shape)
        print(batch['rgbs'].shape)
        break
    for i, batch in tqdm(enumerate(test_loader), total=len(test_set)):
        print(batch['cam'].shape)
        print(batch['rgbs'].shape)
        break

if __name__ == '__main__':
    main()