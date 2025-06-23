import os
import cv2
from tqdm import tqdm
import numpy as np
import struct
from argparse import ArgumentParser
from PIL import Image
import PIL
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
height=480
width=640
height_low=384
trans_totensor_resize = transforms.Compose([
    transforms.Resize((height,width), interpolation=PIL.Image.BILINEAR),
])
trans_totensor_cropped = transforms.Compose([
    transforms.Resize((height,width), interpolation=PIL.Image.BILINEAR),
    transforms.CenterCrop(height),
    transforms.Resize(height_low, interpolation=PIL.Image.BILINEAR),
])

def check_scenes(scenes_list):
    for scene_id in scenes_list:
        scene_path = os.path.join(scans_path, scene_id)
        all_filename_list = os.listdir(scene_path + '/sensor_data')
        rgb_filename_list = []
        pose_filename_list = []
        for file in all_filename_list:
            if 'color' in file:
                rgb_filename_list.append(file)
            elif 'pose' in file:
                pose_filename_list.append(file)
        assert len(rgb_filename_list) == len(pose_filename_list), "the numbers of pose files and images in the scene %s are not equal"%(scene_id)


if __name__ == "__main__":
    parser = ArgumentParser("process rgb images and poses")
    parser.add_argument("--data_path", type=str, required=True, help='root path to scannet dataset')
    parser.add_argument("--out_path", type=str, required=True, help='path to save processed results')
    parser.add_argument("--scene_id", type=str, default='', help='name of one scene')
    parser.add_argument("--scene_list", type=str, default='', help='path to the scene list file')
    parser.add_argument("--frame_interval", type=int, default=1, help='frame interval for data sampling')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The path {data_path} does not exist')
    
    scans_path = os.path.join(data_path, 'scans')
    if not os.path.exists(scans_path):
        raise ValueError(f'The path {scans_path} does not exist')
    
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

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
    
    # check scenes
    check_scenes(scenes_list)
    
    # --------------------------------------process each scene
    for scene_id in tqdm(scenes_list):
        print('processing %s'%(scene_id))
        scene_path = os.path.join(scans_path, scene_id)
        out_scene_path = os.path.join(out_path, scene_id)
        os.makedirs(out_scene_path, exist_ok=True)

        # get frame ids in the scene
        all_filename_list = os.listdir(scene_path + '/sensor_data')
        frame_id_list = []
        for file in all_filename_list:
            if 'color' in file:
                frame_id = file.split('.')[0].split('-')[-1]
                frame_id_list.append(frame_id)
        frame_id_list.sort()

        # load and process sampled data
        frame_id_list_sampled = frame_id_list[::args.frame_interval]

        # ---------------------------------------------- step 1: process poses
        c2w_list = []
        frame_id_list_new = []
        frame_id_list_sampled_ = []
        for i in range(len(frame_id_list_sampled)):
            frame_id = frame_id_list_sampled[i]
            pose_filename = f'frame-{frame_id}.pose.txt'
            c2w = np.loadtxt(os.path.join(scene_path, 'sensor_data', pose_filename))
            if not np.isinf(c2w).any():
                c2w_list.append(c2w)
                frame_id_list_new.append('%06d'%(i))
                frame_id_list_sampled_.append(frame_id)
        pose_all_unscaled = np.stack(c2w_list, axis=0)  # n, 4, 4
        frame_id_list_sampled = frame_id_list_sampled_
        
        # save poses
        os.makedirs(os.path.join(out_scene_path, 'pose_unnormalized'), exist_ok=True)
        for i in range(len(pose_all_unscaled)):
            # save unscaled (unnormalized) pose
            np.savetxt(os.path.join(out_scene_path, "pose_unnormalized", "%s.txt"%(frame_id_list_new[i])), pose_all_unscaled[i])
        
        # ---------------------------------------------- step 2: process intrinsic
        # load intrinsic
        intrinsic_path = os.path.join(scene_path, 'intrinsic', 'intrinsic_depth.txt')
        intrinsic_ori = np.loadtxt(intrinsic_path)[:3, :3]

        os.makedirs(os.path.join(out_scene_path, 'intrinsic'), exist_ok=True)
        np.savetxt(os.path.join(out_scene_path, 'intrinsic', 'intrinsic_depth_high.txt'), intrinsic_ori)

        # ---------------------------------------------- step 3: process rgb image
        os.makedirs(os.path.join(out_scene_path, 'image_high'), exist_ok=True)  # for planarSplat
        for i in tqdm(range(len(frame_id_list_sampled))):
            frame_id = frame_id_list_sampled[i]
            rgb_filename = f'frame-{frame_id}.color.jpg'
            rgb_ori = Image.open(os.path.join(scene_path, 'sensor_data', rgb_filename))
            rgb_tensor_resized = trans_totensor_resize(rgb_ori)
            rgb_tensor_resized.save(os.path.join(out_scene_path, 'image_high', '%s.png'%(frame_id_list_new[i])))