import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import open3d as o3d

from utils.model_util import get_K_Rt_from_P
from utils.mesh_util import refuse_mesh
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.mesh_util import render_depth
import math
from loguru import logger
from typing import NamedTuple, List, Dict
import trimesh

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int

class ViewInfo(nn.Module):
    def __init__(self, cam_info: Dict, gt_info: Dict):
        super().__init__()
        # get cam info
        self.intrinsic = cam_info['intrinsic'].cuda()
        self.pose = cam_info['pose'].cuda()
        self.raster_cam_w2c = cam_info['raster_cam_w2c'].cuda()
        self.raster_cam_proj = cam_info['raster_cam_proj'].cuda()
        self.raster_cam_fullproj = cam_info['raster_cam_fullproj'].cuda()
        self.raster_cam_center = cam_info['raster_cam_center'].cuda()
        self.raster_cam_FovX = cam_info['raster_cam_FovX'].cpu().item()
        self.raster_cam_FovY = cam_info['raster_cam_FovY'].cpu().item()
        self.tanfovx = math.tan(self.raster_cam_FovX  * 0.5)
        self.tanfovy = math.tan(self.raster_cam_FovY * 0.5)
        self.raster_img_center = cam_info['raster_img_center'].cuda()
        self.cam_loc = cam_info['cam_loc'].cuda()

        # get gt info
        self.rgb = gt_info['rgb'].cuda()
        self.mono_depth = gt_info['mono_depth'].cuda()
        self.mono_normal_local = gt_info['mono_normal_local'].cuda()
        self.mono_normal_global = gt_info['mono_normal_global'].cuda()
        self.index = gt_info['index']
        self.image_path = gt_info['image_path']

        # other info
        self.scale = 1.0
        self.shift = 0.0
        self.plane_depth = None

class SceneDatasetDemo:
    def __init__(
        self,
        data,
        img_res: List,
        dataset_name: str = 'demo',
        scan_id: str = 'example',
        scene_bounding_sphere: float = 5.0,
        pre_align: bool = False,
        voxel_length: float=0.05,
        sdf_trunc: float=0.08,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.scan_id = scan_id
        self.scene_bounding_sphere = scene_bounding_sphere
        assert self.scene_bounding_sphere > 0.
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res  # [height, width]

        image_paths = data['image_paths']
        self.n_images = len(image_paths)

        # load camera
        self.intrinsics_all = [torch.from_numpy(intrinsic).cuda() for intrinsic in data['intrinsics']]
        self.poses_all = [torch.from_numpy(extrinsic).cuda() for extrinsic in data['extrinsics']]
        
        # load rgbs
        rgbs = [torch.from_numpy(rgb).cuda().float()/255. for rgb in data['color']]
        rgbs = torch.stack(rgbs, dim=0).contiguous()  # n, h, w, 3
        assert rgbs.shape[0] == self.n_images
        assert rgbs.shape[1] == img_res[0]
        assert rgbs.shape[2] == img_res[1]
        rgbs = rgbs.reshape(self.n_images, -1, 3)  # n, hw, 3

        # load depths
        mono_depths = [torch.from_numpy(depth).cuda() for depth in data['depth']]
        mono_depths = torch.stack(mono_depths, dim=0).contiguous()   # n, h, w
        assert mono_depths.shape[0] == self.n_images
        assert mono_depths.shape[1] == img_res[0]
        assert mono_depths.shape[2] == img_res[1]
        mono_depths[mono_depths > 2.0 * self.scene_bounding_sphere] = 0.
        mono_depths = mono_depths.reshape(self.n_images, -1)  # n, hw

        # load normals
        mono_normals = [torch.from_numpy(normal).cuda()*2.-1. for normal in data['normal']]
        mono_normals = torch.stack(mono_normals, dim=0).permute(0, 2, 3, 1).contiguous()  # n, h, w, 3
        assert mono_normals.shape[0] == self.n_images
        assert mono_normals.shape[1] == img_res[0]
        assert mono_normals.shape[2] == img_res[1]
        mono_normals = mono_normals.reshape(self.n_images, -1, 3)  # n, hw, 3


        mesh = refuse_mesh(
            [x.cpu().squeeze().reshape(img_res[0], img_res[1]).numpy() for x in mono_depths],
            [x.cpu().numpy() for x in self.poses_all],
            [x.cpu().numpy() for x in self.intrinsics_all],
            img_res[0],
            img_res[1],
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc)
        absolute_img_path = os.path.abspath(image_paths[0])
        current_dir = os.path.dirname(absolute_img_path)
        parent_dir = os.path.dirname(current_dir)
        self.mono_mesh_dest = os.path.join(parent_dir, 'mono_mesh.ply')
        o3d.io.write_triangle_mesh(self.mono_mesh_dest, mesh)     
        if pre_align:
            mesh = trimesh.load_mesh(self.mono_mesh_dest)
            rendered_depths = render_depth(mesh, [pose.cpu().numpy() for pose in self.poses_all], [intrinsic.cpu().numpy()[:3, :3] for intrinsic in self.intrinsics_all], H=img_res[0], W=img_res[1])
            rendered_depths = [torch.from_numpy(rd).reshape(-1).float() for rd in rendered_depths]
            from utils.align import align_depth_scale
            for i in tqdm(range(len(mono_depths)), desc='aligning depth...'):
                md = mono_depths[i].cuda()
                rd = rendered_depths[i].cuda()
                weight = ((md - rd).abs() <= 0.5) & (rd > 0.05)
                d_scale = align_depth_scale(md.reshape(1, -1), rd.reshape(1, -1), weight=weight.reshape(1, -1).float())
                if d_scale > 0:
                    md = md * d_scale.item()
                mono_depths[i] = md

        # get cam parameters for rasterization
        self.raster_cam_w2c_list, self.raster_cam_proj_list, self.raster_cam_fullproj_list, self.raster_cam_center_list, self.raster_cam_FovX_list, self.raster_cam_FovY_list, self.raster_img_center_list = self.get_raster_cameras(
            self.intrinsics_all, self.poses_all, img_res[0], img_res[1])
        
        # prepare view list
        self.view_info_list = []
        for idx in tqdm(range(self.n_images), desc='building view list...'):
            cam_loc = self.poses_all[idx][:3, 3].clone()            
            cam_info = {
                "intrinsic": self.intrinsics_all[idx].clone(),
                "pose": self.poses_all[idx].clone(),  # camera to world
                "raster_cam_w2c": self.raster_cam_w2c_list[idx].clone(),
                "raster_cam_proj": self.raster_cam_proj_list[idx].clone(),
                "raster_cam_fullproj": self.raster_cam_fullproj_list[idx].clone(),
                "raster_cam_center": self.raster_cam_center_list[idx].clone(),
                "raster_cam_FovX": self.raster_cam_FovX_list[idx].clone(),
                "raster_cam_FovY": self.raster_cam_FovY_list[idx].clone(),
                "raster_img_center": self.raster_img_center_list[idx].clone(),
                "cam_loc": cam_loc.squeeze(0),
            }

            normal_local = mono_normals[idx].clone().cuda()
            normal_global = normal_local @ self.poses_all[idx][:3, :3].T

            gt_info = {
                "rgb": rgbs[idx],
                "image_path": image_paths[idx],
                "mono_depth": mono_depths[idx],
                "mono_normal_global": normal_global,
                "mono_normal_local": normal_local,
                'index': idx
            }
            self.view_info_list.append(ViewInfo(cam_info, gt_info))      

            

        logger.info('data loader finished')
    
    def load_cameras(self, cam_dict, n_images, debug_start_idx=-1):
        if debug_start_idx == -1:
            scale_mats = [cam_dict['scale_mat_%d' % idx].to(dtype=torch.float32) for idx in range(n_images)]
            world_mats = [cam_dict['world_mat_%d' % idx].to(dtype=torch.float32) for idx in range(n_images)]
        else:
            scale_mats = [cam_dict['scale_mat_%d' % (debug_start_idx + idx)].to(dtype=torch.float32) for idx in range(n_images)]
            world_mats = [cam_dict['world_mat_%d' % (debug_start_idx + idx)].to(dtype=torch.float32) for idx in range(n_images)]

        intrinsics_all = []
        poses_all = []

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, pose = get_K_Rt_from_P(None, P.numpy())
            intrinsics_all.append(torch.from_numpy(intrinsic).float().cuda())
            poses_all.append(torch.from_numpy(pose).float().cuda())
        
        return intrinsics_all, poses_all
    
    def get_raster_cameras(self, intrinsics_all, poses_all, height, width):
        zfar = 10.
        znear = 0.01
        raster_cam_w2c_list = []
        raster_cam_proj_list = []
        raster_cam_fullproj_list = []
        raster_cam_center_list = []
        raster_cam_FovX_list = []
        raster_cam_FovY_list = []
        raster_img_center_list = []

        for i in range(self.n_images):
            focal_length_x = intrinsics_all[i][0,0]
            focal_length_y = intrinsics_all[i][1,1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

            cx = intrinsics_all[i][0, 2]
            cy = intrinsics_all[i][1, 2]

            c2w = poses_all[i]  # 4, 4
            w2c = c2w.inverse()  # 4, 4
            w2c_right = w2c.T

            world_view_transform = w2c_right.clone()
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            raster_cam_w2c_list.append(world_view_transform)
            raster_cam_proj_list.append(projection_matrix)
            raster_cam_fullproj_list.append(full_proj_transform)
            raster_cam_center_list.append(camera_center)
            raster_cam_FovX_list.append(torch.tensor([FovX]).cuda())
            raster_cam_FovY_list.append(torch.tensor([FovY]).cuda())

            raster_img_center_list.append(torch.tensor([cx, cy]).cuda())
        
        return raster_cam_w2c_list, raster_cam_proj_list, raster_cam_fullproj_list, raster_cam_center_list, raster_cam_FovX_list, raster_cam_FovY_list, raster_img_center_list
