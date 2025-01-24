

import numpy as np
import torch
import random
from collections import defaultdict
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import open3d

def cameraplane2world_tensor(planedetect, Tcw):
    return Tcw.t() @ planedetect


def generate_spare_point_cloud(depth_map, m, f, cx, cy):
    H, W = depth_map.shape
    point_cloud = []
    dict_depth_point = {}
    for _ in range(m):
        h = random.randint(0, H - 1)
        w = random.randint(0, W - 1)
        Z = depth_map[h, w]
        if Z == 0:
            continue
        # to world coordinate
        X = (h - cx) * Z / f
        Y = (w - cy) * Z / f
        point_cloud.append((X, Y, Z))
        dict_depth_point[str([h, w, depth_map[h, w]])] = np.array([X, Y, Z, 1]).reshape((4, 1))

    return point_cloud, dict_depth_point


def ransac(pointcloud, first=False, dataset="replica"):
    distance_threshold = 0.01# inner to planes max dis
    ransac_n = 3
    if dataset == "replica":
        num_iterations = 1500
        if first:
            num_iterations = 3000
            points_in_plane_thread = 150
            num_points_rest = 200
        else:
            points_in_plane_thread = 150
            num_points_rest = 200
    elif dataset == "scannet":
        num_iterations = 1800
        if first:
            num_iterations = 2500
            points_in_plane_thread = 250
            num_points_rest = 300
        else:
            points_in_plane_thread = 250
            num_points_rest = 300
    src_points = pointcloud
    src = src_points
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(src)
    number = src_points.shape[0]
    out = pcd
    plane_para = []
    cloudsnum = []
    while number > num_points_rest:
        para, inliers = out.segment_plane(distance_threshold, ransac_n, num_iterations)
        distance_threshold *= 0.9
        inlier_cloud = out.select_by_index(inliers)
        if len(inliers) > points_in_plane_thread:
            cloudsnum.append(len(inlier_cloud.points))
            plane_para.append(para)
        out = out.select_by_index(inliers, invert=True)
        number = len(out.points)
    return plane_para, cloudsnum
def as_intrinsics_matrix(intrinsics):

    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    return K

def get_rays_from_uv_singlepose(i, j, c2w, H, W, fx, fy, cx, cy, device):

    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d
def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):

    # Get pdf
    # weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    pdf = weights

    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples

def select_samples( H, W, samples):

    # indice = torch.randint(H*W, (samples,))
    indice = random.sample(range(H * W), int(samples))
    indice = torch.tensor(indice)
    return indice
def random_select(l, k):

    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device,return_ij=False):

    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1)# pixel frame to word frame这是世界坐标系下的视角方向,想转换为世界坐标下的具体点需要再乘以深度值Z
    dirs_unsque = dirs.unsqueeze(-2)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs_unsque * c2ws[:, None, :3, :3], -1)
    rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)
    if return_ij:
        return rays_o, rays_d, dirs
    else:

        return rays_o, rays_d

def select_uv(i, j, n, b, depths, colors, device='cuda:0'):

    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n * b,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n * b)
    j = j[indices]  # (n * b)

    indices = indices.reshape(b, -1)
    i = i.reshape(b, -1)
    j = j.reshape(b, -1)

    depths = depths.reshape(b, -1)
    colors = colors.reshape(b, -1, 3)

    depths = torch.gather(depths, 1, indices)  # (b, n)
    colors = torch.gather(colors, 1, indices.unsqueeze(-1).expand(-1, -1, 3))  # (b, n, 3)

    return i, j, depths, colors

def get_sample_uv(H0, H1, W0, W1, n, b, depths, colors, device='cuda:0'):

    depths = depths[:, H0:H1, W0:W1]
    colors = colors[:, H0:H1, W0:W1]

    i, j = torch.meshgrid(torch.linspace(W0, W1 - 1, W1 - W0, device=device), torch.linspace(H0, H1 - 1, H1 - H0, device=device))

    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, b, depths, colors, device=device)

    return i, j, depth, color

def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2ws, depths, colors, device,return_ij=False):

    b = c2ws.shape[0]
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, b, depths, colors, device=device)


    if return_ij:
        rays_o, rays_d,dirs = get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device, return_ij)
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), sample_depth.reshape(-1), sample_color.reshape(-1, 3),i,j,dirs
    else:
        rays_o, rays_d = get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device, return_ij)
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), sample_depth.reshape(-1), sample_color.reshape(-1, 3)

def matrix_to_cam_pose(batch_matrices, RT=True):

    if RT:
        return torch.cat([matrix_to_quaternion(batch_matrices[:,:3,:3]), batch_matrices[:,:3,3]], dim=-1)
    else:
        return torch.cat([batch_matrices[:, :3, 3], matrix_to_quaternion(batch_matrices[:, :3, :3])], dim=-1)

def cam_pose_to_matrix(batch_poses):

    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]

    return c2w

def get_rays(H, W, fx, fy, cx, cy, c2w, device):

    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d
def find_planes(normals, counts, angle_threshold=20):#find base 1-3 planes in the first frame
    cos_threshold = np.cos(np.radians(angle_threshold))

    planes = defaultdict(list)
    for i, normal in enumerate(normals):
        for plane_normal in planes.keys():
            cos_angle = np.dot(normal[:3], plane_normal[:3]) / (np.linalg.norm(normal[:3]) * np.linalg.norm(plane_normal[:3]))
            if cos_angle >= cos_threshold:
                planes[plane_normal].append((counts[i], i))
                break
        else:
            planes[tuple(normal)].append((counts[i], i))
    result = []
    result_index_in_origin=[]
    index=0
    for plane_group in planes.values():
        plane_group.sort(reverse=True)
        _, max_index = plane_group[0]
        result.append(normals[max_index])
        result_index_in_origin.append(max_index)
        index=index+1
        if index==3:
            break

    return result,result_index_in_origin

def normalize_3d_coordinate(p, bound):

    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p
def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                          torch.arange(H, dtype=torch.float32), indexing='xy')

    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type is 'OpenGL':  # 这里应该是nice-slam提到的pytorch的坐标系好像是方向反着的
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)
    elif type is 'OpenCV':
        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d