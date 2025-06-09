import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

import numpy as np
import os
import glob
from tqdm import tqdm
import open3d as o3d

def run_vggt(img_dir, out_path, ckpt_path='checkpoints/model.pt', step=1, depth_conf_thresh=2.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if isinstance(img_dir, list):
        img_name_list = img_dir[::step]
    elif os.path.isdir(img_dir):
        img_name_list = glob.glob(img_dir + '/*')
        img_name_list = sorted(img_name_list)[::step]
    else:
        raise ValueError(f'The input img_dir {img_dir} should be either a list which consist of image paths or a directory which contain images.')
    # Load and preprocess example imagess
    images = load_and_preprocess_images(img_name_list).to(device)

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    # model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model = VGGT()
    local_ckpt_path = ckpt_path
    model.load_state_dict(torch.load(local_ckpt_path))
    model = model.to(device)

    image_paths_list = []
    color_images_list = []
    depth_maps_list = []
    c2ws_list = []
    intrinsics_list = []

    min_depth_threshold = 0.2
    median_depth_threshold = 2.0

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

            pose_enc = predictions['pose_enc']  # 1, N, 9
            # Extrinsic (1, N, 3, 4; w2c) and intrinsic (1, N, 3, 3) matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            c2w = closed_form_inverse_se3(extrinsic[0])  # N, 4, 4

            depth = predictions['depth'][0].squeeze(-1)  # N, h, w

            depth_conf = predictions['depth_conf'][0]  # N, h, w
            point_map_by_unprojection = unproject_depth_map_to_point_map(depth.squeeze(0)[..., None], extrinsic.squeeze(0), intrinsic.squeeze(0))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_map_by_unprojection.reshape(-1, 3)[::20])
            pcd_path = os.path.join(out_path, 'pcd_vggt_downsampled.ply')
            o3d.io.write_point_cloud(pcd_path, pcd)

            scale  = 1.0
            depth_masked = depth * (depth_conf>depth_conf_thresh).float()

            depth_median_ = torch.median(depth_masked[depth_masked > 0.0]).cpu().item()
            scale = median_depth_threshold / depth_median_
            depth = depth * scale
            c2w[:,:3,3] = c2w[:,:3,3] * scale
            print(f'scale = {scale}')

            for i in tqdm(range(depth.shape[0]), total=depth.shape[0]):
                depth_i = ((depth_conf[i]>depth_conf_thresh).float() * depth[i]).detach().cpu().squeeze().numpy()
                color_i = (images[i] * 255).cpu().numpy().astype(np.uint8).transpose(1,2,0)                

                image_paths_list.append(img_name_list[i])
                color_images_list.append(color_i)
                depth_maps_list.append(depth_i)
                c2ws_list.append(c2w[i].cpu().numpy())
                intrinsics_list.append(intrinsic[0, i].cpu().numpy())

    return {
        'color': color_images_list,
        'depth': depth_maps_list,
        'image_paths': image_paths_list,
        'extrinsics': c2ws_list,  # c2w
        'intrinsics': intrinsics_list,
    }