# This file is modified from ContraNeRF
import torch
import torch.nn.functional as F


class Projector():
    def __init__(self):
        pass

    def inbound(self, pixel_locations, h, w):
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).type_as(pixel_locations).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, src_cams):
        B, Nv = src_cams.shape[:2]
        src_intrinsics = src_cams[..., 2:18].reshape(B*Nv, 4, 4)[:, :3, :3]  # [B*n_views, 3, 3]
        src_poses = src_cams[..., -16:].reshape(B*Nv, 4, 4) 
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [B, n_points, 4]
        projections = src_intrinsics.bmm(
            torch.inverse(src_poses).bmm(xyz_h.transpose(-1, -2).repeat(Nv, 1, 1))[:, :3]
            )  # [B*n_views, 3, n_points]
        projections = projections.transpose(-2, -1).reshape(B, Nv, -1, 3)  # [B, n_views, n_points, 3]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [B, n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e5, max=1e5)
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera
        return pixel_locations, mask

    def compute_angle(self, xyz, cam, src_cams):
        B, Np, _ = xyz.shape
        Nv = src_cams.shape[1]
        src_poses = src_cams[..., -16:].reshape(B, Nv, 4, 4)
        query_pose = cam[:, -16:].reshape(B, 1, 4, 4).expand(-1, Nv, -1, -1)  # [B, n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :, :3, 3].unsqueeze(2) - xyz.unsqueeze(1)) # [B, n_views, n_samples, 3]
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2src_pose = (src_poses[:, :, :3, 3].unsqueeze(2) - xyz.unsqueeze(1))
        ray2src_pose = ray2src_pose / (torch.norm(ray2src_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2src_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2src_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1) # [B, n_views, n_samples, 4]
        return ray_diff

    def compute(self, xyz, cam, src_imgs, src_cams, featmaps):
        r"""
        Input:
            xyz: [B, n_samples, 3]
            cam: [B, 1, 34]
            src_imgs: [B, n_views, 3, h, w]
            src_cams: [B, n_views, 34]
            featmaps: [B, n_views, d, h1, w1]
        Output:
            rgb_feat_sampled: [B, n_samples, n_views, d+3]
            ray_diff: [B, n_samples, n_views, 4]
            mask: [B, n_samples, n_views, 1]
        """
        B, Nv, _, H, W = src_imgs.shape
        Np = xyz.shape[1]
        # xyz = xyz.reshape(B, 128, 128, 96, 3)[:, 20:80, 30:90].reshape(B, -1, 3)

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front = self.compute_projections(xyz, src_cams) # [B, n_views, n_samples, 2], [B, n_views, n_samples]
        # avoid numerical precision errors
        pixel_locations[(pixel_locations < 0) & (pixel_locations > -0.5)] = 0
        pixel_locations[(pixel_locations > H - 1) & (pixel_locations < H - 0.5)] = H - 1
        pixel_locations[(pixel_locations > W - 1) & (pixel_locations < W - 0.5)] = W - 1
        
        # # visualize for debug
        # import matplotlib.pyplot as plt

        # painted_img  = src_imgs.clone().detach().cpu()[0].numpy()
        # pixel_locations[(pixel_locations.abs() > 500).any(-1)] = -100
        # for v in range(Nv):
        #     plt.imshow(painted_img[v, :, :, :].transpose(1, 2, 0))
        #     plt.scatter(pixel_locations[0, v, :, 0].cpu().numpy(), pixel_locations[0, v, :, 1].cpu().numpy(), c='r')
        #     plt.savefig('debug/{}.png'.format(v))
        #     plt.close()
        normalized_pixel_locations = self.normalize(pixel_locations, H, W).reshape(B*Nv, 1, -1, 2)  # [B*n_views, 1, n_samples, 2]

        # rgb sampling
        src_imgs = src_imgs.flatten(0, 1)  # [B*n_views, 3, h, w]
        rgbs_sampled = F.grid_sample(src_imgs, normalized_pixel_locations, align_corners=True).view(B, Nv, 3, Np) # [B, n_views, 3, n_samples]
        rgbs_sampled = rgbs_sampled.permute(0, 3, 1, 2) # [B, n_samples, n_views, 3]

        # deep feature sampling
        featmaps = featmaps.flatten(0, 1)  # [B*n_views, d, h1, w1]
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True).view(B, Nv, -1, Np) # [B, n_views, d, n_samples]
        feat_sampled = feat_sampled.permute(0, 3, 1, 2) # [B, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat([rgbs_sampled, feat_sampled], dim=-1)   # [B, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, H, W)
        # ray_diff = self.compute_angle(xyz, cam, src_cams)
        # ray_diff = ray_diff.permute(0, 2, 1, 3)
        mask = (inbound * mask_in_front).float().transpose(1, 2)[..., None]   # [B, n_samples, n_views, 1]
        return rgb_feat_sampled, mask, pixel_locations