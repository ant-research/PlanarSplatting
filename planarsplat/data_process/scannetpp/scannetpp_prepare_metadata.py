import os
from colmap_io import read_extrinsics_text, read_intrinsics_text
from colmap_io import qvec2rotmat
import numpy as np
import shutil
from argparse import ArgumentParser

parser = ArgumentParser("process rgb images and poses")
parser.add_argument("--data_path", type=str, required=True, help='root path to scannet dataset')
parser.add_argument("--out_root_path", type=str, required=True, help='path to save processed results')
parser.add_argument("--scene_id", type=str, default='', help='name of one scene')
parser.add_argument("--scene_list", type=str, default='', help='path to the scene list file')
args = parser.parse_args()

data_path = args.data_path
if not os.path.exists(data_path):
    raise ValueError(f'The path {data_path} does not exist')

out_root_path = args.out_root_path
os.makedirs(out_root_path, exist_ok=True)

# get scene ids
if args.scene_id:
    scenes_list = [args.scene_id]
elif args.scene_list:
    scenes_list = []
    with open(args.scene_list, 'r') as file:
        lines = file.readlines()
        for line in lines:
            cleaned_line = line.strip()
            scenes_list.append(cleaned_line)
else:
    raise ValueError("One of the [scene_id, scene_list] should be given!")
        
for scene in scenes_list:
    print(f'processing {scene}')
    rgb_path = os.path.join(data_path, scene, 'iphone/rgb')
    if not os.path.exists(rgb_path): 
        continue
    
    rgb_frame_names = os.listdir(rgb_path)
    if len(rgb_frame_names) == 0:
        continue
    
    colmap_cam_file_path = os.path.join(data_path, scene, 'iphone/colmap/cameras.txt')
    if not os.path.exists(colmap_cam_file_path):
        continue
    colmap_image_file_path = os.path.join(data_path, scene, 'iphone/colmap/images.txt')
    if not os.path.exists(colmap_image_file_path):
        continue

    cameras = read_intrinsics_text(colmap_cam_file_path)
    camera = next(iter(cameras.values()))
    fx, fy, cx, cy = camera.params[:4]
    intrinsic_out_path = os.path.join(out_root_path, scene, 'intrinsic')
    os.makedirs(intrinsic_out_path, exist_ok=True)
    intrinsic_color = np.array([[fx, 0., cx, 0.],
                                [0.,fy, cy, 0.],
                                [0.,0.,1.0,0.],
                                [0.,0.,0.,1.0]])
    np.savetxt(os.path.join(intrinsic_out_path, 'intrinsic_color.txt'), intrinsic_color, fmt='%.6f')

    hr = 480 / camera.height
    wr = 640 / camera.width
    fx_ = fx * wr
    fy_ = fy * hr
    cx_ = cx * wr
    cy_ = cy * hr
    intrinsic_depth = np.array([[fx_, 0., cx_, 0.],
                                [0.,fy_, cy_, 0.],
                                [0.,0.,1.0,0.],
                                [0.,0.,0.,1.0]])
    np.savetxt(os.path.join(intrinsic_out_path, 'intrinsic_depth.txt'), intrinsic_depth, fmt='%.6f')

    images_meta = read_extrinsics_text(colmap_image_file_path)

    i = 0
    for img_id, img_meta in images_meta.items():
        image_meta = img_meta
        image_id = image_meta.id
        frame_name = image_meta.name
        q = image_meta.qvec
        t = image_meta.tvec
        r = qvec2rotmat(q)
        rt = np.eye(4)
        rt[:3,:3] = r
        rt[:3, 3] = t
        c2w = np.linalg.inv(rt)

        out_path = os.path.join(out_root_path, scene, 'sensor_data')
        os.makedirs(out_path, exist_ok=True)
        shutil.copy2(os.path.join(rgb_path, frame_name), os.path.join(out_path, 'frame-%06d.color.jpg'%(i)))

        np.savetxt(os.path.join(out_path, 'frame-%06d.pose.txt'%(i)), c2w, fmt='%.6f')
        i += 1

    