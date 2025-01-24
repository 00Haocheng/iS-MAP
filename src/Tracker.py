import torch
import copy
import os
import time
from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples,cameraplane2world_tensor,generate_spare_point_cloud,ransac)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer

class Tracker(object):
    """
    Tracking main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """
    def __init__(self, cfg, args, ismap):
        self.cfg = cfg
        self.args = args
        self.scale = cfg['scale']
        self.idx = ismap.idx
        self.bound = ismap.bound
        self.mesher = ismap.mesher
        self.output = ismap.output
        self.verbose = ismap.verbose
        self.renderer = ismap.renderer
        self.gt_c2w_list = ismap.gt_c2w_list
        self.mapping_idx = ismap.mapping_idx
        self.mapping_cnt = ismap.mapping_cnt
        self.shared_decoders = ismap.shared_decoders
        self.estimate_c2w_list = ismap.estimate_c2w_list
        self.truncation = ismap.truncation
        self.shared_higher_planes_xy = ismap.shared_higher_planes_xy
        self.shared_higher_planes_xz = ismap.shared_higher_planes_xz
        self.shared_higher_planes_yz = ismap.shared_higher_planes_yz
        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.keyframe_every=cfg['mapping']['keyframe_every']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=True, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = ismap.H, ismap.W, ismap.fx, ismap.fy, ismap.cx, ismap.cy

        self.decoders = copy.deepcopy(self.shared_decoders)

        self.higher_planes_xy = copy.deepcopy(self.shared_higher_planes_xy)
        self.higher_planes_xz = copy.deepcopy(self.shared_higher_planes_xz)
        self.higher_planes_yz = copy.deepcopy(self.shared_higher_planes_yz)

        for p in self.decoders.parameters():
            p.requires_grad_(False)
        self.plane_mask=ismap.plane_mask
        self.struct_para = ismap.struct_para

    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values:  [n_rays,40]每条射线采样了40个点，zvals就是它们的深度,40是采样的点数
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values 每条射线对应的深度真值

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss
        return sdf_losses

    def optimize_tracking(self, cam_pose, gt_color, gt_depth, gt_c2w,batch_size, optimizer,iters,planes_detected):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        all_planes = (self.higher_planes_xy, self.higher_planes_xz, self.higher_planes_yz)
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c2w = cam_pose_to_matrix(cam_pose)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
                                                                   self.device, self.truncation, gt_depth=batch_gt_depth)

        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)
        ## SDF losses
        loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
        ## Color Loss
        color_loss=self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()
        loss = loss + color_loss
        ### Depth loss
        depth_loss=self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()
        loss=loss+depth_loss
        #Manhattan matching loss
        if  planes_detected!=None and iters>self.num_cam_iters/2:
            Tcw = c2w.squeeze()
            planes_detected_world = cameraplane2world_tensor(planes_detected, Tcw)
            Parassociate_thread = 0.965
            vertical_thread = 0.17
            planes_detected_world_num = planes_detected_world.shape[1]
            base_plane0 = self.base_plane[0][:3]
            base_plane1 = self.base_plane[1][:3]
            base_plane2 = self.base_plane[2][:3]
            planefeature_loss = []
            vertical_planefeature_loss=[]
            for ii in range(planes_detected_world_num):
                oneplane = planes_detected_world[:3, ii]
                angle0 = abs(oneplane @ base_plane0)
                if angle0 > Parassociate_thread:
                    error_vector = 1 - angle0
                    planefeature_loss.append(error_vector)
                if angle0 < vertical_thread:
                    error_vector = angle0
                    vertical_planefeature_loss.append(error_vector)

                if base_plane1[0]!=0:
                    angle1 = abs(oneplane @ base_plane1)
                    if angle1 > Parassociate_thread:
                        error_vector = 1-angle1
                        planefeature_loss.append(error_vector)
                    if angle1 < vertical_thread:
                        error_vector = angle1
                        vertical_planefeature_loss.append(error_vector)

                if base_plane2[0] != 0:
                    angle2 = abs(oneplane @ base_plane2)
                    if angle2 > Parassociate_thread:
                        error_vector = 1 - angle2
                        planefeature_loss.append(error_vector)
                    if angle2 < vertical_thread:
                        error_vector = angle2
                        vertical_planefeature_loss.append(error_vector)

            if len(planefeature_loss)>0:
                planefeature_lossv = torch.stack(planefeature_loss)
                planefeature_losses = torch.mean(planefeature_lossv, dim=0)
                loss = loss + planefeature_losses * 0.2

            if len(vertical_planefeature_loss) > 0:
                verticalplanefeature_lossv = torch.stack(vertical_planefeature_loss)
                verticalplanefeature_losses = torch.mean(verticalplanefeature_lossv, dim=0)
                loss = loss +  verticalplanefeature_losses*0.02

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def update_params_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

            self.decoders.load_state_dict(self.shared_decoders.state_dict())

            for higher_planes, self_higher_planes in zip(
                    [self.shared_higher_planes_xy, self.shared_higher_planes_xz, self.shared_higher_planes_yz],
                    [self.higher_planes_xy, self.higher_planes_xz, self.higher_planes_yz]):
                for i, higher_plane in enumerate(higher_planes):
                    self_higher_planes[i] = higher_plane.detach()
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        """
            Runs the tracking thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        device = self.device
        all_planes = (self.higher_planes_xy, self.higher_planes_xz, self.higher_planes_yz)

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w,line_seg in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depthsque = gt_depth.squeeze(0)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

            self.update_params_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                self.base_plane = self.struct_para #Manhattan base plane
                if not self.no_vis_on_first_frame:
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)

            else:
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
                else:
                    ## Initialize with the last known pose
                    cam_pose = matrix_to_cam_pose(pre_c2w)
                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                         {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])
                current_min_loss = torch.tensor(float('inf')).float().to(device)
                gtseg=None
                planes_detected=None
                if idx % self.every_frame== 0 and idx!=0:
                    depth_numpy = gt_depthsque.numpy()
                    if self.cfg['dataset']=="scannet":
                        t=80
                    elif self.cfg['dataset']=='replica':
                        t = 200
                    #generate spare pointcloud for faster speed
                    pointcloud, dict_depth_point = generate_spare_point_cloud(depth_numpy, int(self.W * self.H / t),
                                                                              self.fx, self.cx, self.cy)
                    pointcloud = np.array(pointcloud).reshape((-1, 3))

                    # planes segmentation
                    mask_rgb = np.zeros(depth_numpy.shape).astype(np.uint8)
                    #ransac plane param(Hessian form)
                    planes_detected, clouds= ransac(pointcloud,dataset=self.cfg['dataset'])
                    print("use %d planes in %d frame" % (len(planes_detected), idx))
                    for key in dict_depth_point.keys():
                        for i, plane in enumerate(planes_detected):
                            if abs(np.dot(plane, dict_depth_point[key])) < 0.01:
                                coordinates = eval(str(key))
                                mask_rgb[coordinates[0], coordinates[1]] = i + 1
                    #pass the plane segmentation to mapper
                    gtseg = torch.from_numpy(mask_rgb).to(self.device)
                    self.plane_mask[:,:]=gtseg.clone()

                    for n, pla in enumerate(planes_detected):
                        if pla[3] < 0:
                            planes_detected[n] = pla * -1

                    planes_detected_tensorlist = [torch.from_numpy(arr) for arr in planes_detected]
                    if len(planes_detected) > 0:
                        planes_detected = torch.stack(planes_detected_tensorlist, dim=1).float().to(device)
                    else:
                        planes_detected = None
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)
                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)
                    trackloss_view = torch.abs(matrix_to_cam_pose(gt_c2w).to(device) - cam_pose).mean().item()
                    loss = self.optimize_tracking(cam_pose, gt_color, gt_depth, gt_c2w,self.tracking_pixels, optimizer_camera,cam_iter,planes_detected)
                    if cam_iter == self.num_cam_iters - 1:
                        print(f"rend loss:{loss:.4f}" + f"    camera loss {trackloss_view:.4f}")
                    if cam_iter>self.num_cam_iters/2+1:
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_pose = cam_pose.clone().detach()
                    candidate_cam_pose = cam_pose.clone().detach()
                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx