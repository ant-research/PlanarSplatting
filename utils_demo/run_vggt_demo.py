import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

import argparse
import glob
import random
import string
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument("--data_path", type=str, default='examples/rooms_vggt/images', help='path of input data')
args = parser.parse_args()

data_path = args.data_path
if not os.path.exists(data_path):
    raise ValueError(f'The input path {data_path} does not exist.')

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


img_name_list = glob.glob(data_path + '/*')
img_name_list = sorted(img_name_list)
# Load and preprocess example imagess
images = load_and_preprocess_images(img_name_list).to(device)


# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model = VGGT()
local_ckpt_path = "checkpoints/model.pt"
model.load_state_dict(torch.load(local_ckpt_path))
model = model.to(device)

characters = string.ascii_letters + string.digits
out_scene_id = ''.join(random.choice(characters) for _ in range(6))

out_path_img = f'outputs/{out_scene_id}/images'
out_path_depth = f'outputs/{out_scene_id}/depths'
os.makedirs(out_path_img, exist_ok=False)
os.makedirs(out_path_depth, exist_ok=False)

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

        print(f"saving results into outputs/{out_scene_id}...")
        for i in tqdm(range(depth.shape[0]), total=depth.shape[0]):
            # save depth            
            save_path = os.path.join(out_path_depth, f'{i:06}.depth.png')
            depth_i = ((depth_conf[i]>3.0).float() * depth[i]).detach().cpu().squeeze().numpy()
            plt.imsave(save_path, depth_i, cmap='viridis')
            # np.save('')
            # save points
            # points = point_map_by_unprojection[i].reshape(-1, 3)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # save_path = os.path.join(out_path, f'tmp_{i}.ply')
            # o3d.io.write_point_cloud(save_path, pcd)

            # save rgb image
            rgb_origin = (images[i] * 255).cpu().numpy().astype(np.uint8).transpose(1,2,0)
            save_path = os.path.join(out_path_img, f'{i:06}.color.png')
            cv2.imwrite(save_path, cv2.cvtColor(rgb_origin, cv2.COLOR_RGB2BGR))

            # save pose (c2w)

            # save intrinsic

