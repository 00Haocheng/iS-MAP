import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d
import random
import os
import time
import numpy as np
from colorama import Fore, Style
import cv2
from src.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix,get_rays_from_uv_singlepose,select_samples,matrix_to_quaternion,find_planes,ransac,generate_spare_point_cloud)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.tools.cull_mesh import cull_mesh


class Mapper(object):
    def __init__(self, cfg, args, ismap):

        self.cfg = cfg
        self.args = args

        self.idx = ismap.idx
        self.truncation = ismap.truncation
        self.bound = ismap.bound
        self.logger = ismap.logger
        self.mesher = ismap.mesher
        self.output = ismap.output
        self.verbose = ismap.verbose
        self.renderer = ismap.renderer
        self.mapping_idx = ismap.mapping_idx
        self.mapping_cnt = ismap.mapping_cnt
        self.decoders = ismap.shared_decoders

        self.higher_planes_xy = ismap.shared_higher_planes_xy
        self.higher_planes_xz = ismap.shared_higher_planes_xz
        self.higher_planes_yz = ismap.shared_higher_planes_yz

        self.estimate_c2w_list = ismap.estimate_c2w_list
        self.gt_c2w_list = ismap.gt_c2w_list
        self.mapping_first_frame = ismap.mapping_first_frame

        self.scale = cfg['scale']
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.joint_opt = False  # Even if joint_opt is enabled, it starts only when there are at least 4 keyframes
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr'] # The learning rate for camera poses during mapping
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.every_frame = cfg['mapping']['every_frame']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.plane_mask=ismap.plane_mask  #plane segmentation
        self.struct_para = ismap.struct_para #base plane
        self.keyframe=ismap.keyframeDatabase
        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=True,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = ismap.H, ismap.W, ismap.fx, ismap.fy, ismap.cx, ismap.cy

    def coordinates(self,voxel_dim, device: torch.device, flatten=True):
        if type(voxel_dim) is int:
            nx = ny = nz = voxel_dim
        else:
            nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
        x = torch.arange(0, nx, dtype=torch.long, device=device)
        y = torch.arange(0, ny, dtype=torch.long, device=device)
        z = torch.arange(0, nz, dtype=torch.long, device=device)
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")

        if not flatten:
            return torch.stack([x, y, z], dim=-1)

        return torch.stack((x.flatten(), y.flatten(), z.flatten()))

    def smoothness(self, allplanes, sample_points=64, voxel_size=0.1, margin=0.05, color=False):
        volume = self.bound[:, 1] - self.bound[:, 0]

        grid_size = (sample_points - 1) * voxel_size
        offset_max = self.bound[:, 1] - self.bound[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = self.coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1, 1, 1, 3)).to(volume)) * voxel_size + self.bound[:, 0] + offset

        query_points = (pts - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])

        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]]).to(self.device)

        embedded =self.decoders.hash_embed_fn(inputs_flat)
        add_feature_plane_smooth=False
        if add_feature_plane_smooth:
            higher_planes_xy, higher_planes_xz, higher_planes_yz = allplanes
            embedded2 = self.decoders.sample_plane_feature(inputs_flat, higher_planes_xy, higher_planes_xz, higher_planes_yz)
            embedded = torch.cat([embedded, embedded2], dim=-1)
        sdf= torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])
        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()
        loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)
        return loss
    def sdf_losses(self, sdf, z_vals, gt_depth):
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))#sdf-1
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))
        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss
        return sdf_losses

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0),gt_color.unsqueeze(0), device)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(1, -1, 3)#

        keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in self.keyframe_list], dim=0)
        w2cs = torch.inverse(keyframes_c2ws[:-2])

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        ## Considering only overlapped frames
        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes

    def compute_line_loss(self, depth_image, se_image, allplanes, c2w, device):
        K_re = torch.eye(3)
        K_re[0, 0] = self.fx
        K_re[1, 2] = self.cy
        K_re[1, 1] = self.fy
        K_re[0, 2] = self.cx
        K_re_inv = torch.inverse(K_re).to(device)
        f = se_image.squeeze()
        num_struct_keysets =128
        keyset_samples = 3
        f_flat = f.flatten()
        f_flat = f_flat.cpu().numpy()
        f = f.cpu().numpy()
        num_struct_pixels = np.sum(f_flat > 0)
        if num_struct_pixels < 800:
            return 0
        else:
            num_struct = f_flat.max()
            acc_num = 0
            pix_i = []
            pix_j = []
            batch_gt_depth = []
            num_seg_pixel = [0] * (num_struct + 1)
            for j in range(num_struct):
                pixels_j = np.sum(f_flat == j + 1)
                num_j = int(np.ceil(num_struct_keysets * pixels_j / num_struct_pixels))
                acc_num += num_j
                num_seg_pixel[j + 1] = num_j

                inx_j = np.argwhere(f == j + 1)
                keyset_j = inx_j[np.random.randint(inx_j.shape[0], size=num_j * keyset_samples)]
                for m in keyset_j:
                    Us, Vs = m
                    Z = depth_image[Us, Vs]
                    pix_i.append(Us)
                    pix_j.append(Vs)
                    batch_gt_depth.append(Z)
            pix_j = torch.Tensor(pix_j).to(device)
            pix_i = torch.Tensor(pix_i).to(device)

            rays_o, rays_d = get_rays_from_uv_singlepose(pix_i, pix_j, c2w, self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                                              device)
            batch_rays_o,batch_rays_d=rays_o.squeeze(),rays_d.squeeze()
            batch_gt_depth = torch.Tensor(batch_gt_depth).squeeze()

            batch_gt_depth = batch_gt_depth.to(device)

            depth_plane= self.renderer.render_batch_ray(allplanes, self.decoders, batch_rays_d,
                                                                       batch_rays_o, device, self.truncation,
                                                                       gt_depth=batch_gt_depth,onlydepth=True)

            #convert to world coordinate
            U2=torch.stack([pix_i,pix_j,torch.ones_like(pix_i)])

            p2=torch.matmul( K_re_inv,U2)
            keypoints=p2*depth_plane
            xielulist=[]
            for v in range(keypoints.shape[1]-1):
                xielu=keypoints[:,v]-keypoints[:,v+1]
                xielu=xielu/((xielu[0]**2+xielu[1]**2+xielu[2]**2)**0.5)
                xielulist.append(xielu)
            start_points = keypoints[:, ::3]
            end_points_A = keypoints[:, 1::3]
            end_points_B = keypoints[:, 2::3]
            vector_A = end_points_A - start_points
            vector_B = end_points_B - start_points
            AxB = torch.cross(vector_A, vector_B, dim=0)
            line_loss = torch.norm(AxB,p=2,dim=0).mean()

            return line_loss
    def compute_plane_loss(self,depth_image, se_image, allplanes, c2w, device):
        K_re = torch.eye(3)
        K_re[0, 0] = self.fx
        K_re[1, 2] = self.cy
        K_re[1, 1] = self.fy
        K_re[0, 2] = self.cx
        K_re_inv = torch.inverse(K_re).to(device)
        f = se_image
        num_struct_keysets = 256  # select 256 pairs plane pixels in one seg image
        keyset_samples = 4  # one pair have 4 pixels
        f_flat = f.flatten()
        f_flat = f_flat.cpu().numpy()
        f = f.cpu().numpy()
        num_struct_pixels = np.sum(f_flat > 0)

        if num_struct_pixels < 1110:  # too few plane seg pixels
            return 0

        num_struct = f_flat.max()  # how many classes in the plane seg image
        acc_num = 0
        pix_i = []
        pix_j = []
        batch_gt_depth = []
        num_seg_pixel = [0] * (num_struct + 1)
        for j in range(num_struct):
            pixels_j = np.sum(f_flat == j + 1)
            num_j = int(np.ceil(num_struct_keysets * pixels_j / num_struct_pixels)) #how much pair pixel are selected in j plane
            acc_num += num_j
            num_seg_pixel[j + 1] = num_j
            inx_j = np.argwhere(f == j + 1)  # j plane pixel indexes in the seg image
            keyset_j = inx_j[np.random.randint(inx_j.shape[0], size=num_j * keyset_samples)]#random select 4*num_j pixels in j plane
            for m in keyset_j:
                Us, Vs = m
                Z = depth_image[Us, Vs]
                pix_i.append(Us)
                pix_j.append(Vs)
                batch_gt_depth.append(Z)
        pix_j = torch.Tensor(pix_j).to(device)
        pix_i = torch.Tensor(pix_i).to(device) # pix_i ,pxi_j means the selected plane pixel index in the image

        rays_o, rays_d = get_rays_from_uv_singlepose(pix_i, pix_j, c2w, self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                                          device)  # from the index generate rays

        batch_rays_o,batch_rays_d=rays_o.squeeze(),rays_d.squeeze()
        batch_gt_depth = torch.Tensor(batch_gt_depth).squeeze()
        batch_gt_depth = batch_gt_depth.to(device)
        depth_plane= self.renderer.render_batch_ray(allplanes, self.decoders, batch_rays_d,
                                                                   batch_rays_o, device, self.truncation,
                                                                   gt_depth=batch_gt_depth,onlydepth=True)

        #convert to 3D camera coordinate
        U2=torch.stack([pix_i,pix_j,torch.ones_like(pix_i)])

        p2=torch.matmul( K_re_inv,U2)
        keypoints=p2*depth_plane
        # Four points is a group ;
        start_points = keypoints[:, ::4]
        end_points_A = keypoints[:, 1::4]
        end_points_B = keypoints[:, 2::4]
        end_points_C = keypoints[:, 3::4]
        vector_A = end_points_A - start_points
        vector_B = end_points_B - start_points
        vector_C = end_points_C - start_points
        AxB = torch.cross(vector_A, vector_B, dim=0)
        AxB_dot_C = torch.sum(AxB * vector_C, dim=0)
        plane_loss = torch.abs(AxB_dot_C).mean()

        return plane_loss

    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w,keyframe_dict, keyframe_list, planeseg,cur_c2w,lineseg):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if joint_opt enables).

        Args:
            iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): a list of dictionaries of keyframes info.
            keyframe_list (list): list of keyframes indices.
            cur_c2w (tensor): the estimated camera to world matrix of current frame.
            planeseg(tensor): segmentation of plane
            lineseg(tensor): segmentation of line

        Returns:
            cur_c2w: return the updated cur_c2w, return the same input cur_c2w if no joint_opt
        """
        all_planes = (self.higher_planes_xy,self.higher_planes_xz,self.higher_planes_yz)
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.keyframe_dict)-2, self.mapping_window_size-1)
            elif self.keyframe_selection_method == 'overlap':
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, self.mapping_window_size-1)

        # add the last two keyframes and the current frame(use -1 to denote)
        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list)-1] + [len(keyframe_list)-2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        decoders_para_list += list(self.decoders.parameters())
        higher_planes_para = []
        # set plane learnable
        for higher_planes in [self.higher_planes_xy, self.higher_planes_xz, self.higher_planes_yz]:
            for i, higher_plane in enumerate(higher_planes):
                higher_plane = nn.Parameter(higher_plane)
                higher_planes_para.append(higher_plane)
                higher_planes[i] = higher_plane
        gt_depths = []
        gt_colors = []
        c2ws = []
        gt_c2ws = []
        for frame in optimize_frame:
            # the oldest frame should be fixed to avoid drifting
            if frame != -1:
                gt_depths.append(keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(keyframe_dict[frame]['color'].to(device))
                c2ws.append(keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])
            else:
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)

        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        c2ws = torch.stack(c2ws, dim=0)

        if self.joint_opt:
            cam_poses = nn.Parameter(matrix_to_cam_pose(c2ws[1:]))

            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': higher_planes_para, 'lr': 0},
                                          {'params': [cam_poses], 'lr': 0}])


        else:
            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': higher_planes_para, 'lr': 0}
                                          ])

        optimizer.param_groups[0]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
        optimizer.param_groups[1]['lr'] = cfg['mapping']['lr']['higher_planes_lr'] * lr_factor

        if self.joint_opt:
            optimizer.param_groups[2]['lr'] = self.joint_opt_cam_lr

        for joint_iter in range(iters):
            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.decoders)

            if self.joint_opt:
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(cam_poses)], dim=0)
            else:
                c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,index_i,index_j,dirs = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, device,return_ij=True)
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(
                    device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth

            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders,batch_rays_d,
                                                                       batch_rays_o, device, self.truncation,
                                                                       gt_depth=batch_gt_depth)
            depth_mask = (batch_gt_depth > 0)
            # SDF losses
            loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

            color_loss = self.w_color * torch.square(batch_gt_color - color).mean()
            ## Color loss
            loss = loss + color_loss
            ## Depth loss
            loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

            # smooth loss
            if idx != 0 and self.decoders.use_hash:

                loss_smooth = self.smoothness(all_planes)
                if self.cfg['dataset'] == "scannet":
                    k_smooth = 0.25
                elif self.cfg['dataset'] == 'replica':
                    k_smooth = 0.01
                loss_smooth = loss_smooth * k_smooth
                loss +=  loss_smooth

            #structural consistence loss
            loss_plane, loss_line = 0, 0
            if joint_iter > 1 * iters / 2:
                if self.cfg['dataset'] == "scannet":
                    kplane, kline = 0.05, 0.005
                elif self.cfg['dataset'] == 'replica':
                    kplane, kline = 0.05, 0.05
                loss_plane = kplane * self.compute_plane_loss(cur_gt_depth, planeseg, all_planes, cur_c2w, device)  # for plane consis
                loss = loss_plane + loss
                loss_line = kline*self.compute_line_loss(cur_gt_depth, lineseg, all_planes, cur_c2w, device)
                loss = loss_line  + loss
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        if self.joint_opt:
            # put the updated camera poses back
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())
            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]
        return cur_c2w
    def cameraplane2world(self,planedetect, Tcw):
        worldplanes = []
        for plane_coe in planedetect:
            worldplan = Tcw.transpose() @ plane_coe
            distance = (worldplan[0] ** 2 + worldplan[1] ** 2 + worldplan[2] ** 2) ** 0.5
            worldplan = worldplan / distance
            worldplanes.append(worldplan)
        return worldplanes

    def run(self):
        cfg = self.cfg
        all_planes = (self.higher_planes_xy,self.higher_planes_xz,self.higher_planes_yz)
        idx, gt_color, gt_depth, gt_c2w,segline_img = self.frame_reader[0]
        data_iterator = iter(self.frame_loader)

        ## Fixing the first camera pose
        self.estimate_c2w_list[0] = gt_c2w

        init_phase = True
        prev_idx = -1#the last mapping frame idx
        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1: ## Last input frame
                    break

                if idx % self.every_frame == 0 and idx != prev_idx:
                    break

                time.sleep(0.001)

            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w, segline_img= next(data_iterator)
            gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
            gt_depthsque = gt_depth.squeeze(0)
            gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

            gt_depth = gt_depthsque.to(self.device, non_blocking=True)

            cur_c2w = self.estimate_c2w_list[idx]

            if not init_phase:
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                iters = cfg['mapping']['iters_first']

            ## Deciding if camera poses should be jointly optimized
            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            if idx== 0:
                #plane segmentation and create base planes
                depth_numpy = gt_depthsque.numpy()
                if self.cfg['dataset'] == "scannet":
                    t = 70
                elif self.cfg['dataset'] == 'replica':
                    t = 200
                # generate spare pointcloud for faster speed
                pointcloud, dict_depth_point = generate_spare_point_cloud(depth_numpy, int(self.W * self.H / t),
                                                                          self.fx, self.cx, self.cy)
                pointcloud = np.array(pointcloud).reshape((-1, 3))

                # planes segmentation
                mask_rgb = np.zeros(depth_numpy.shape).astype(np.uint8)
                # ransac plane param(Hessian form)
                planes_detected, cloudsnum = ransac(pointcloud, first=True,dataset=self.cfg['dataset'])  # 假设planes_detected是一个n*4的列表
                #find 1-3 base planes param
                if self.cfg['dataset'] == "scannet":
                    planes_detected,planes_index = find_planes(planes_detected, cloudsnum, angle_threshold=10)
                elif self.cfg['dataset'] == 'replica':
                    planes_detected,planes_index = find_planes(planes_detected, cloudsnum)

                for key in dict_depth_point.keys():
                    for i, plane in enumerate(planes_detected):
                        if abs(np.dot(plane, dict_depth_point[key])) < 0.01:
                            coordinates = eval(str(key))
                            mask_rgb[coordinates[0], coordinates[1]] = i + 1  # 将在平面上的点在掩码中对应的位置设为平面的索引

                Tcw = cur_c2w.cpu().numpy()
                for n, pla in enumerate(planes_detected):
                    if pla[3] < 0:
                        planes_detected[n] = pla * -1
                planes_detected_world = self.cameraplane2world(planes_detected, Tcw)
                planes_detected_world_numpy = np.array(planes_detected_world)
                planes_detected_tensor=torch.from_numpy(planes_detected_world_numpy)

                #Pass the base plane to tracker
                if len(planes_detected) == 1:
                    self.struct_para[ 0, :] = planes_detected_tensor.to(self.device).clone()
                elif len(planes_detected) == 2:
                    self.struct_para[ :2, :] = planes_detected_tensor.to(self.device).clone()
                else:
                    self.struct_para[:,:] = planes_detected_tensor.to(self.device).clone()
                segplane_img = torch.from_numpy(mask_rgb).to(self.device)
            else:
                segplane_img=self.plane_mask.clone()
            cur_c2w = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w,
                                            self.keyframe_dict, self.keyframe_list, segplane_img,cur_c2w,segline_img.to(self.device))

            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w
            # add new frame to keyframe set
            if idx % self.keyframe_every == 0:
                self.keyframe_list.append(idx)
                self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color.to(self.keyframe_device),
                                           'depth': gt_depth.to(self.keyframe_device), 'est_c2w': cur_c2w.clone()})

            init_phase = False
            self.mapping_first_frame[0] = 1     # mapping of first frame is done, can begin tracking

            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img-1:
                self.logger.log(idx, self.keyframe_list,all_planes)
            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list[:idx+1])

            if idx == self.n_img-1:

                if self.eval_rec:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                else:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'

                self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)

            if idx == self.n_img-1:
                break