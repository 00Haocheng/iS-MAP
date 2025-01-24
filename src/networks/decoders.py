import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate
from src.networks.encode import get_hash_encoder

class Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta

    """
    def __init__(self, c_dim=36, hidden_size=64, hidden_hidden_size=32,truncation=0.08, n_blocks=2, learnable_beta=True,use_hash=True,use_sdffeature=True,bound=None):
        super().__init__()
        self.c_dim = c_dim
        self.truncation = truncation
        self.n_blocks = n_blocks
        self.use_hash=use_hash
        self.bound=bound
        self.use_sdffeatur=use_sdffeature
        self.linears = nn.ModuleList(
            [nn.Linear(c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
        self.c_linears = nn.ModuleList(
            [nn.Linear(c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_hidden_size)  for i in range(n_blocks - 1)])
        if self.use_hash:
            dim_max = (self.bound[:, 1] - self.bound[:, 0]).max()
            voxel_sdf=0.02#for replica
            # voxel_sdf = 0.04  # for scannet
            resolution_sdf = int(dim_max / voxel_sdf)
            self.hash_embed_fn, self.hash_en_ch = get_hash_encoder("HashGrid", log2_hashmap_size=16,desired_resolution=resolution_sdf)
            self.linears = nn.ModuleList(
                [nn.Linear(c_dim + self.hash_en_ch, hidden_size)] +
                [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_size, 1)
        if use_sdffeature:
            self.sdffeature_dim = 32
            self.output_linear = nn.Linear(hidden_size, 1 + self.sdffeature_dim)
            self.c_linears = nn.ModuleList(
                [nn.Linear(c_dim + +self.sdffeature_dim, hidden_size)] +
                [nn.Linear(hidden_size, hidden_hidden_size) for i in range(n_blocks - 1)])
        
        self.c_output_linear = nn.Linear(hidden_hidden_size, 3)

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """

        vgrid = p_nor[None, :, None] #p_nor:[n_rays*n_stratified+n , 3]
        feat_new = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            feat_new.append([xy,xz,yz])
        feat_new0 = torch.cat(feat_new[0], dim=-1)
        feat_new1 = torch.cat(feat_new[1], dim=-1)
        feat_new2 = torch.cat(feat_new[2], dim=-1)
        feat_new3 = torch.cat(feat_new[3], dim=-1)
        feat_new=torch.cat((feat_new0, feat_new1,feat_new2,feat_new3),dim=1)
        return feat_new 


    def get_raw_sdf(self, p_nor, all_planes):
        """
        Get raw SDF
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            sdf (tensor): raw SDF
        """
        higher_planes_xy, higher_planes_xz, higher_planes_yz = all_planes

        feat = self.sample_plane_feature(p_nor, higher_planes_xy, higher_planes_xz, higher_planes_yz)
        if self.use_hash:
            inputs3 = self.hash_embed_fn(p_nor)
            feat = torch.cat([feat, inputs3], dim=-1)
        h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        if self.use_sdffeatur:
            sdfandfeature = torch.tanh(self.output_linear(h)).squeeze()
            sdf, geo_feat = sdfandfeature[..., :1], sdfandfeature[..., 1:]
            return sdf,geo_feat
        else:
            sdf= torch.tanh(self.output_linear(h)).squeeze()
            return sdf
    def get_raw_rgb(self, p_nor, all_planes,geofeature=None):
        """
        Get raw RGB
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            rgb (tensor): raw RGB
        """
        higher_planes_xy, higher_planes_xz, higher_planes_yz = all_planes
        
        c_feat = self.sample_plane_feature(p_nor, higher_planes_xy, higher_planes_xz, higher_planes_yz)
        if self.use_sdffeatur:
            c_feat = torch.cat([geofeature,c_feat], dim=-1)
        h = c_feat
        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb

    def forward(self, p, all_planes,onlydepth=False):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            raw (tensor): raw SDF and RGB将给定的三维坐标p进行归一化处理，使得坐标值落在[-1, 1]的范围内，并返回归一化后的结果
        """
        p_shape = p.shape#[n_rays, n_stratified+n_importance, 3]

        p_nor = normalize_3d_coordinate(p.clone(), self.bound)#[n_rays*n_stratified+n , 3]
        if onlydepth:
            if self.use_sdffeatur:
                sdf, geofeature= self.get_raw_sdf(p_nor, all_planes)
            else:
                sdf = self.get_raw_sdf(p_nor, all_planes)
            raw=sdf.unsqueeze(-1)
            raw = raw.reshape(*p_shape[:-1], -1)

            return raw
        else:
            if self.use_sdffeatur:
                sdf,geofeature = self.get_raw_sdf(p_nor, all_planes)
                rgb = self.get_raw_rgb(p_nor, all_planes, geofeature)
            else:
                sdf= self.get_raw_sdf(p_nor, all_planes)
                sdf=sdf.unsqueeze(-1)
                rgb = self.get_raw_rgb(p_nor, all_planes)

            raw = torch.cat([rgb, sdf], dim=-1)
            raw = raw.reshape(*p_shape[:-1], -1)
            return raw
