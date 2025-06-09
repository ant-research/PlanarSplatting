import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import open3d as o3d
import numpy as np
from tqdm import tqdm
import numpy as np
from typing import List
import glob
import torch

def refuse_mesh(depths: List[np.ndarray], 
                poses: List[np.ndarray], 
                intrinsics: List[np.ndarray], 
                H: int, 
                W: int, 
                voxel_length: float=0.05, 
                sdf_trunc: float=0.08, 
                depth_trunc: float=5.0
                ):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for pose, K, depth_pred in tqdm(zip(poses, intrinsics, depths)):
        depth_pred = depth_pred.reshape(H, W)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

def get_coarse_mesh(net, 
                    view_info_list: List, 
                    H: int, 
                    W: int, 
                    voxel_length: float=0.05, 
                    sdf_trunc: float=0.08
                    ):
    scene_scale = net.planarSplat.pose_cfg.scale
    scene_offset = net.planarSplat.pose_cfg.offset
    poses = []
    intrinsics = []
    for view_info in view_info_list:
        pose = view_info.pose.clone()
        pose[:3, 3] /= scene_scale
        pose[:3, 3] += torch.tensor(scene_offset).to(pose.device)
        poses.append(pose.cpu().numpy())
        intrinsics.append(view_info.intrinsic[:3, :3].cpu().numpy())

    depths = []
    for iter in range(len(view_info_list)):
        with torch.no_grad():
            allmap = net.planarSplat(view_info_list[iter], 50000)
        # get rendered maps
        depth = allmap[0:1].squeeze().reshape(H, W).cpu().numpy() / scene_scale
        depths.append(depth)

    mesh = refuse_mesh(depths, poses, intrinsics, H, W, voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    return mesh