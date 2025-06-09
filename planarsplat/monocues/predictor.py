import glob
import os
import numpy as np
import cv2
import torch
from .metric3d import metric3d_convnext_tiny, metric3d_convnext_large, metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
from tqdm import tqdm
import sys
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
    )
)
from utils import mesh_util
import open3d as o3d

class MonoCuesPredictor:
    '''
    MonoCuesPredictor is a class wrapper of Metric3D (and omnidata)that predicts the depth and normal of a scene that consists of multiple calibrated RGB images. 
    We assume the intrinsics are known for each image.
    '''
    depth_inference_size = (616, 1064) # for metric3d_vit_large
    def __init__(self, 
                 scene_dir: str, 
                 image_folder: str = None,
                 depth_model_name: str = 'metric3d_vit_large',
                 image_ext: str = 'jpg',
                 intrinsic_relpath: str = 'intrinsic/intrinsic_depth.txt',
                 pose_relpath: str = 'pose_unnormalized/*.txt',
                 voxel_length: float = 0.05,
                 sdf_trunc: float = 0.08,
                 sample_interval: int = 1,
                 ):
        self.setup_models(depth_model_name)
        self.scene_dir = scene_dir
        self.intrinsic_relpath = intrinsic_relpath
        self.pose_relpath = pose_relpath
        if image_folder is None:
            self.image_files = sorted(glob.glob(os.path.join(scene_dir, f'*.{image_ext}')))
        else:
            self.image_files = sorted(glob.glob(os.path.join(scene_dir, image_folder, f'*.{image_ext}')))
        self.pose_files = sorted(glob.glob(os.path.join(scene_dir, pose_relpath)))[::sample_interval]
        self.image_files = self.image_files[::sample_interval]

        self.poses = torch.stack([torch.from_numpy(np.loadtxt(pose_file)) for pose_file in self.pose_files])


        with open(os.path.join(scene_dir, intrinsic_relpath), 'r') as f:
            self.intrinsic_matrix = np.loadtxt(f)

        self.mono_dest = os.path.join(scene_dir, f'mono_{image_folder}_{depth_model_name}.pth')
        self.mono_mesh_dest = os.path.join(scene_dir, f'mono_mesh_{image_folder}_{depth_model_name}.ply')
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        
    
    def setup_models(self, depth_model_name: str):
        self.depth_model_name = depth_model_name
        self.depth_model = self.get_depth_model(depth_model_name).to('cuda')
        
    def get_depth_model(self, depth_model_name: str):
        if depth_model_name == 'metric3d_convnext_tiny':
            model = metric3d_convnext_tiny(pretrain=True)
        elif depth_model_name == 'metric3d_convnext_large':
            model = metric3d_convnext_large(pretrain=True)
        elif depth_model_name == 'metric3d_vit_small':
            model = metric3d_vit_small(pretrain=True)
        elif depth_model_name == 'metric3d_vit_large':
            model = metric3d_vit_large(pretrain=True)
        elif depth_model_name == 'metric3d_vit_giant2':
            model = metric3d_vit_giant2(pretrain=True)
        else:
            raise ValueError(f'Invalid depth model name: {depth_model_name} (should be one of [metric3d_convnext_tiny, metric3d_convnext_large, metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2])')
        
        if depth_model_name == 'metric3d_convnext_tiny' or depth_model_name == 'metric3d_convnext_large':
            self.depth_inference_size = (544, 1216)

        return model
        
    def predict_depth_batch(self, batch_size: int = 4, overwrite: bool = False):
        if os.path.exists(self.mono_dest) and not overwrite:
            print(f'MonoCuesPredictor: {self.mono_dest} already exists, skipping...')
            return torch.load(self.mono_dest, weights_only=False)
        
        cameras = {}
        for i, pose  in enumerate(self.poses):
            cameras['scale_mat_%d'%(i)] = torch.eye(4, dtype=torch.float64)
            K = torch.eye(4,dtype=torch.float64)
            K[:3, :3] = torch.from_numpy(self.intrinsic_matrix)
            cameras['world_mat_%d'%(i)] = K@torch.linalg.inv(pose)

        rgb_origin = [cv2.imread(image_file)[:, :, ::-1] for image_file in self.image_files]
        h,w = rgb_origin[0].shape[:2]
        
        assert all([(x.shape[0]==h) and (x.shape[1]==w)] for x in rgb_origin), "The size of every rgb image should be equal"
        
        rgb_tensor = []
        rgb_origin_tensor = []
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

        scale = min(self.depth_inference_size[0] / h, self.depth_inference_size[1] / w)
        h, w = int(h*scale), int(w*scale)
        pad_h = self.depth_inference_size[0] - h 
        pad_w = self.depth_inference_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        intrinsic = [
                self.intrinsic_matrix[0, 0]*scale, 
                self.intrinsic_matrix[1, 1]*scale, 
                self.intrinsic_matrix[0, 2]*scale, 
                self.intrinsic_matrix[1, 2]*scale
        ]
        
        for rgb in rgb_origin:
            
            rgb_i = cv2.resize(rgb, 
                               (w,h),
                               interpolation=cv2.INTER_LINEAR
                                )

            rgb_i = cv2.copyMakeBorder(rgb_i, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean.flatten().tolist())
            rgb_i = torch.from_numpy(rgb_i.transpose((2, 0, 1))).float()
            rgb_i = torch.div((rgb_i - mean), std)
            rgb_tensor.append(rgb_i)

            rgb_origin_i = np.ascontiguousarray(rgb.transpose((2, 0, 1)))
            rgb_origin_tensor.append(torch.from_numpy(rgb_origin_i).float() / 255.)  # 3, H, W

        rgb_tensor = torch.stack(rgb_tensor, dim=0)
        rgb_origin_tensor = torch.stack(rgb_origin_tensor, dim=0)
        
        depth_list = []
        normal_list = []

        for i, tensor in enumerate(tqdm(rgb_tensor.split(batch_size,dim=0), desc='Predicting depth and normal')):
            with torch.no_grad():
                pred_depth, confidence, output_dict = self.depth_model.inference({'input': tensor.to('cuda')})
            pred_normal = output_dict['prediction_normal']

            pred_depth = pred_depth[:, :, pad_info[0]:pred_depth.shape[2]-pad_info[1], pad_info[2]:pred_depth.shape[3]-pad_info[3]]
            pred_normal = pred_normal[:, :, pad_info[0]:pred_normal.shape[2]-pad_info[1], pad_info[2]:pred_normal.shape[3]-pad_info[3]]

            pred_depth = torch.nn.functional.interpolate(pred_depth, rgb_origin[0].shape[:2], mode='bilinear')
            canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
            pred_depth = torch.clamp(pred_depth, 0, 300)

            pred_normal = torch.nn.functional.interpolate(pred_normal, rgb_origin[0].shape[:2], mode='bilinear')
            pred_normal = (pred_normal[:, :3] + 1) / 2.0
            
            depth_list.append(pred_depth.cpu())
            normal_list.append(pred_normal.cpu())

        depth_tensor = torch.cat(depth_list, dim=0)
        normal_tensor = torch.cat(normal_list, dim=0)
        
        mono_dict = {
            'rgb': rgb_origin_tensor,
            'depth': depth_tensor,
            'normal': normal_tensor,
            'image_files': self.image_files,
            'cameras': cameras,
        }

        mesh = mesh_util.refuse_mesh(
            [x.squeeze().numpy() for x in depth_tensor.split(1,dim=0)],[x.numpy() for x in self.poses],
            [self.intrinsic_matrix for _ in range(len(self.poses))],
            rgb_origin[0].shape[0],
            rgb_origin[0].shape[1],
            self.voxel_length,
            self.sdf_trunc)
        
        torch.save(mono_dict, self.mono_dest)
        o3d.io.write_triangle_mesh(self.mono_mesh_dest, mesh)   
        
        return mono_dict

            