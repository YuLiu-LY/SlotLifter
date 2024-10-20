import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image
import hydra
import torch
from glob import glob
import numpy as np
import random
import csv
from util.misc import SubSampler
from util.camera import rotate_cam
from util.ray import get_rays


class MultiscenesDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.split = split
        self.root = f"{cfg.dataset_root}/{cfg.subset}/{cfg.subset}_test" if split != 'train' else f"{cfg.dataset_root}/{cfg.subset}/{cfg.subset}_train"
        self.subset = cfg.subset
        self.n_scenes = 5000
        self.ray_batchsize = cfg.ray_batchsize
        self.img_size = cfg.img_size[0], cfg.img_size[1]
        self.num_src_view = 1
        self.depth_range = [6, 20]
        self.max_depth = 20
        self.render_src_view = cfg.render_src_view
        self.load_mask = cfg.load_mask
        self.norm_scene = cfg.norm_scene
        self.subsampler = SubSampler()
        self.setup_data()
        
    def setup_data(self):
        self.scenes = []
        # for root in self.roots:
        #     self.root = root
        file_path = os.path.join(self.root, 'files.csv')
        if os.path.exists(file_path):
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    self.scenes.append(row)
        else:
            image_filenames = sorted(glob(os.path.join(self.root, '*.png')))  # root/00000_sc000_az00_el00.png
            mask_filenames = sorted(glob(os.path.join(self.root, '*_mask.png')))
            fg_mask_filenames = sorted(glob(os.path.join(self.root, '*_mask_for_moving.png')))
            moved_filenames = sorted(glob(os.path.join(self.root, '*_moved.png')))
            bg_mask_filenames = sorted(glob(os.path.join(self.root, '*_mask_for_bg.png')))
            bg_in_mask_filenames = sorted(glob(os.path.join(self.root, '*_mask_for_providing_bg.png')))
            changed_filenames = sorted(glob(os.path.join(self.root, '*_changed.png')))
            bg_in_filenames = sorted(glob(os.path.join(self.root, '*_providing_bg.png')))
            changed_filenames_set, bg_in_filenames_set = set(changed_filenames), set(bg_in_filenames)
            bg_mask_filenames_set, bg_in_mask_filenames_set = set(bg_mask_filenames), set(bg_in_mask_filenames)
            image_filenames_set, mask_filenames_set = set(image_filenames), set(mask_filenames)
            fg_mask_filenames_set, moved_filenames_set = set(fg_mask_filenames), set(moved_filenames)
            filenames_set = image_filenames_set - mask_filenames_set - fg_mask_filenames_set - moved_filenames_set - changed_filenames_set - bg_in_filenames_set - bg_mask_filenames_set - bg_in_mask_filenames_set
            filenames = sorted(list(filenames_set))
            scenes = []
            for i in range(self.n_scenes):
                scene_filenames = [x for x in filenames if 'sc{:04d}'.format(i) in x]
                if len(scene_filenames) > 0:
                    scenes.append(scene_filenames)
            self.scenes += scenes
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(scenes)
        self.n_scenes = len(self.scenes)

        self.intrinsics = []
        self.scene2normscene = []
        self.normscene_scale = []
        self.cam2normscene = []
        self.cam2scene = []
        img_h, img_w = 256, 256
        for i in tqdm(range(self.n_scenes), desc=f"Loading {self.split} scenes"):
            cam2scene, dims = [], []
            for path in self.scenes[i]:
                pose_path = path.replace('.png', '_RT.txt')
                pose = np.loadtxt(pose_path)
                cam2scene.append(torch.tensor(pose, dtype=torch.float32)) # 4x4
                dims.append([img_h, img_w])
            cam2scene = torch.stack(cam2scene) # N, 4x4
            self.cam2scene.append(cam2scene)

            intrinsic = self.get_intrinsic(path.replace('.png', '_intrinsics.txt'))
            intrinsic_normed = torch.diag(torch.Tensor([self.img_size[1] / img_w, 
                                    self.img_size[0] / img_h, 1, 1])) @ intrinsic
            self.intrinsics.append(intrinsic_normed)
            if self.norm_scene:
                nss_scale = 7 # we follow the setting of uorf
                scene2normscene = torch.Tensor([
                    [1/nss_scale, 0, 0, 0],
                    [0, 1/nss_scale, 0, 0],
                    [0, 0, 1/nss_scale, 0],
                    [0, 0, 0, 1],
                ])
                self.scene2normscene.append(scene2normscene)
                self.normscene_scale.append(scene2normscene[0, 0])
                cam2normscene = []
                indice = list(range(len(cam2scene)))
                for idx in indice:
                    cam2normscene.append(scene2normscene @ cam2scene[idx])
                cam2normscene = torch.stack(cam2normscene)
                self.cam2normscene.append(cam2normscene)
        if self.norm_scene:
            self.depth_range = self.depth_range[0] * min(self.normscene_scale), self.depth_range[1] * max(self.normscene_scale)
        print(f"{self.n_scenes} scenes for {self.split}")
        print(f"depth range: {self.depth_range}")

    def _transform(self, img):
        img = TF.resize(img, self.img_size)
        img = TF.to_tensor(img)
        return img
    
    def get_intrinsic(self, path):
        frustum_size = (256, 256)
        if not os.path.isfile(path):
            focal_ratio = (350. / 320., 350. / 240.)
            focal_x = focal_ratio[0] * frustum_size[0]
            focal_y = focal_ratio[1] * frustum_size[1]
            bias_x = (frustum_size[0] - 1.) / 2.
            bias_y = (frustum_size[1] - 1.) / 2.
        else:
            intrinsics = np.loadtxt(path)
            focal_x = intrinsics[0, 0] * frustum_size[0]
            focal_y = intrinsics[1, 1] * frustum_size[1]
            bias_x = ((intrinsics[0, 2] + 1) * frustum_size[0] - 1.) / 2.
            bias_y = ((intrinsics[1, 2] + 1) * frustum_size[1] - 1.) / 2.
        intrinsic = torch.tensor([[focal_x, 0, bias_x, 0],
                                    [0, focal_y, bias_y, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        return intrinsic.float()

    def _transform_mask(self, img):
        img = TF.resize(img, self.img_size, Image.NEAREST)
        img = TF.to_tensor(img)
        return img
    
    def load_sample(self, scene_idx, sample_idx):
        path = self.scenes[scene_idx][sample_idx]
        img = Image.open(path).convert('RGB')
        img = self._transform(img).permute([1, 2, 0]).view(-1, 3)  # [HW, 3]

        cam2scene = self.cam2normscene[scene_idx][sample_idx] if self.norm_scene else self.cam2scene[scene_idx][sample_idx]
        cam = torch.cat([torch.Tensor(self.img_size), self.intrinsics[scene_idx].flatten(),
                               cam2scene.flatten()])   # 2+16+16=34

        
        return img, cam

    def __getitem__(self, index):
        Br = self.ray_batchsize
        sample = {}
        if self.split == 'train':
            scene_idx = index // 4
            sample_idx = index % 4
            filenames = self.scenes[scene_idx]
            all_rgbs, all_cams = [], []
            for i, path in enumerate(filenames):
                img, cam = self.load_sample(scene_idx, i)
                all_rgbs.append(img)
                all_cams.append(cam)
            all_rgbs = torch.stack(all_rgbs) # [N, H*W, 3]
            all_cams = torch.stack(all_cams) # [N, 34]
            
            tgt_cam = all_cams[0:1] # [1, 34]
            all_rays = get_rays(all_cams, *self.img_size) # [N, HW, 6]
            if np.random.rand() > 0.25:
                # random delete source view
                tgt_rgbs = all_rgbs[self.num_src_view:].view(-1, 3) # [N*Br, 3] # [N-1, HW, 3]
                tgt_rays = all_rays[self.num_src_view:].view(-1, 6) # [N*Br, 6]
            else:
                tgt_rgbs = all_rgbs.view(-1, 3) # [N*Br, 3]
                tgt_rays = all_rays.view(-1, 6) # [N*Br, 6]
            subsample_idx = torch.randperm(tgt_rgbs.shape[0])[:Br][:, None] # [Br]
            tgt_rgbs = tgt_rgbs.gather(0, subsample_idx.expand(-1, 3)) # [Br, 3]
            tgt_rays = tgt_rays.gather(0, subsample_idx.expand(-1, 6)) # [Br, 6]

            src_rgbs = all_rgbs[0:self.num_src_view] # [N, HW, 3]
            src_cams = all_cams[0:self.num_src_view] # [N, 34]
        else:
            filenames = self.scenes[index]
            all_rgbs, all_cams = [], []
            for i, path in enumerate(filenames):
                img, cam = self.load_sample(index, i)
                all_rgbs.append(img)
                all_cams.append(cam)
            all_rgbs = torch.stack(all_rgbs) # [N, H*W, 3]
            all_cams = torch.stack(all_cams) # [N, 34]
            if 'kitchen' in self.subset: # no ground truth mask
                masks = torch.randint(4, [all_rgbs.shape[0], self.img_size[0] * self.img_size[1]])
            else:
                masks = []
                for path in filenames:
                    mask_path = path.replace('.png', '_mask.png')
                    if os.path.isfile(mask_path):
                        mask = Image.open(mask_path).convert('RGB')
                        mask_l = mask.convert('L')
                        mask = self._transform_mask(mask)
                        # ret['mask'] = mask
                        mask_l = self._transform_mask(mask_l)
                        mask_flat = mask_l.flatten(start_dim=0)  # HW,
                        greyscale_dict = mask_flat.unique(sorted=True)  # 8,
                        # make sure the background is always 0
                        if self.subset not in ['room_texture', 'kitchen_shiny', 'kitchen_matte']:
                            bg_color = greyscale_dict[1].clone()
                            greyscale_dict[1] = greyscale_dict[0]
                            greyscale_dict[0] = bg_color
                        onehot_labels = mask_flat[:, None] == greyscale_dict  # HWx8, one-hot
                        onehot_labels = onehot_labels.type(torch.uint8)
                        mask_idx = onehot_labels.argmax(dim=1).view(-1)  # HW
                        masks.append(mask_idx)
                masks = torch.stack(masks) # [N, HW]
            sample['instances'] = masks
            tgt_rgbs, tgt_cam = all_rgbs, all_cams
            tgt_rays = get_rays(tgt_cam, *self.img_size) # [N, HW, 6]
            Nv = self.num_src_view
            src_rgbs, src_cams = all_rgbs[:Nv], all_cams[:Nv]
        sample['rgbs'] = tgt_rgbs # [HW, 3] or [N, HW, 3]
        sample['rays'] = tgt_rays # [HW, 6] or [N, HW, 6]
        sample['cam'] = tgt_cam # [1, 34]  or [N, 34]
        sample['src_rgbs'] = src_rgbs.reshape(-1, *self.img_size, 3) # [N, H, W，3] N = 1
        sample['src_cams'] = src_cams # [N, 34] 2+16+16
        sample['depth_range'] = torch.tensor(self.depth_range).float()

        return sample

    def __len__(self):
        return self.n_scenes if self.split != 'train' else self.n_scenes * 4
    

# test
from torch.utils.data import DataLoader
from tqdm import tqdm
@hydra.main(config_path='../config/cfg', config_name='uorf', version_base='1.2')
def main(config):
    config.img_size = [128, 128]
    config.num_workers = 4
    config.subset = 'kitchen_matte'
    config.dataset_root = "/home/yuliu/Dataset/uorf"
    config.num_src_view = 1
    train_set = MultiscenesDataset(config, 'train')
    val_set = MultiscenesDataset(config, 'val')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    # vis_set = MultiscenesVisualDataset(config)
    # vis_loader = DataLoader(vis_set, batch_size=1, shuffle=False, num_workers=0)
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        rgbs = batch['rgbs']
        print(rgbs.shape)
        cam = batch['cam']
        src_rgbs = batch['src_rgbs']
        src_cams = batch['src_cams']
        if i > 7:
            break
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        rgbs = batch['rgbs']
        mask = batch['instances']
        if i > 5:
            break
    # for i, batch in tqdm(enumerate(vis_loader), total=len(vis_loader)):
    #     cam = batch['cam']
    #     print(cam.shape)
    #     src_rgbs = batch['src_rgbs']
    #     src_cams = batch['src_cams']
    #     if i > 5:
    #         break


if __name__ == '__main__':
    main()

    