# iSMAP is a modified version of https://github.com/idiap/ESLAM
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2023 Johari, M. M. and Carta, C. and Fleuret, F.
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer
from src.utils.keyframe import KeyFrameDatabase


torch.multiprocessing.set_sharing_strategy('file_system')

class iSMAP():
    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args

        self.verbose = cfg['verbose']#realted with if print the output or not
        self.device = cfg['device']
        self.dataset = cfg['dataset']
        self.truncation = cfg['model']['truncation'] #used in model(0.06)

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()# 将输入图像的大小调整为crop_size
        self.scale = cfg['scale']  # 默认为1
        self.load_bound(cfg)
        model = config.get_model(cfg,self.bound)#get the model

        self.shared_decoders = model

        self.init_planes(cfg)
        self.keyframeDatabase = self.create_kf_database(cfg)


        # =======================继续各种类变量初始化操作=======================
        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)

        for shared_higer_planes in [self.shared_higher_planes_xy, self.shared_higher_planes_xz, self.shared_higher_planes_yz]:
            for i, plane in enumerate(shared_higer_planes):
                plane = plane.to(self.device)
                plane.share_memory_()
                shared_higer_planes[i] = plane
        self.shared_decoders = self.shared_decoders.to(self.device)
        self.shared_decoders.share_memory()

        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2w_list.share_memory_()
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()

        self.plane_mask = torch.zeros((self.H, self.W), dtype=torch.uint8, device=self.device)#the plane_segment mask
        self.plane_mask.share_memory_()
        self.struct_para=torch.zeros((3,4),device=self.device)#base planes
        self.struct_para.share_memory_()
        # self.everyframe_para = torch.zeros(( 3, 4), device=self.device)
        # self.everyframe_para.share_memory_()
        # self.everyframe_para_index = torch.zeros(( 3), device=self.device)
        # self.everyframe_para_index.share_memory_()

        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):# 将输入图像的大小调整为crop_size（lietorch 中使用的变量名）
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """

        # scale the bound if there is a global scaling factor 如果存在全局缩放因子，则缩放边界
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float()
        bound_dividable = cfg['planes_res']['bound_dividable']#should be resize to 0.48 or not?
        # enlarge the bound a bit to allow it dividable by bound_dividable
        # 稍微扩大边界以使其可被 bound_divisible 整除
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]
        #这里基本就是读取了重建的边界，然后比读取的稍微扩大了一点点



    def create_kf_database(self, cfg, lenframe=2000,cropsize0=0):
        '''
        Create the keyframe database
        '''
        num_kf = int(lenframe // cfg["mapping"]["every_frame"] + 1)
        print('#kf:', num_kf)
        total_pixel=(cfg['cam']['H'] - cropsize0) * (cfg['cam']['W'] - cropsize0)
        self.num_rays_to_save_inkeyframs=int(cfg["mapping"]["n_pixels"] * total_pixel)
        num_plames=3*num_kf
        print('#Pixels to save:', )
        return KeyFrameDatabase(cfg,
                               self.H,
                                self.W,
                                num_kf,
                                num_plames,
                                self.num_rays_to_save_inkeyframs,
                                self.device)
    def init_planes(self, cfg):
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """
        self.coarse_planes_res = cfg['planes_res']['coarse']
        self.fine_planes_res = cfg['planes_res']['fine']
        self.coarser_planes_res = cfg['planes_res']['coarser']
        self.finer_planes_res = cfg['planes_res']['finer']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]
        higher_planes_xy,  higher_planes_xz,  higher_planes_yz = [], [], []
        planes_res_higher = [self.coarser_planes_res, self.coarse_planes_res, self.fine_planes_res,self.finer_planes_res]#multi-resolution plane
        planes_dim =cfg['model']['each_plane_dim']

        for grid_res in planes_res_higher:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            higher_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            higher_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            higher_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))
        self.shared_higher_planes_xy = higher_planes_xy
        self.shared_higher_planes_xz = higher_planes_xz
        self.shared_higher_planes_yz = higher_planes_yz

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while True:
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread.

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(0, 2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
