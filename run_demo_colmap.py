import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
import numpy as np
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_metric3d import extract_mono_geo_demo
from utils_demo.run_planarSplatting import run_planarSplatting
from utils_demo.read_write_model import read_model
from planarsplat.data_process.scannetpp.colmap_io import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from PIL import Image
import cv2


# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/make_depth_scale.py
def get_scales(key, cameras, images, points3d_ordered, invmonodepthmap):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images_meta[key].point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1

    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", type=str, default='path/to/colmap/data', help='path of input colmap data')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/demo_colmap', help='path of output dir')
    parser.add_argument("--conf_path", type=str, default='utils_demo/demo.conf', help='path of configure file')
    parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    
    image_path = os.path.join(data_path, 'images')
    if not os.path.exists(image_path):
        raise ValueError(f'The input image path {image_path} does not exist.')

    colmap_cam_file_path = os.path.join(data_path, 'sparse/cameras.bin')
    if not os.path.exists(colmap_cam_file_path):
        raise ValueError(f'The input path {colmap_cam_file_path} does not exist.')
    
    colmap_image_file_path = os.path.join(data_path, 'sparse/images.bin')
    if not os.path.exists(colmap_image_file_path):
        raise ValueError(f'The input path {colmap_image_file_path} does not exist.')


    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    precomputed_data_path = os.path.join(out_path, 'data.pth')
    use_precomputed_data = args.use_precomputed_data

    if use_precomputed_data and os.path.exists(precomputed_data_path):
        data = torch.load(precomputed_data_path)
        print(f"loading precomputed data from {precomputed_data_path}")
    else:
        cameras = read_intrinsics_binary(colmap_cam_file_path)
        camera = next(iter(cameras.values()))
        fx, fy, cx, cy = camera.params[:4]
        intrinsic = np.array([[fx, 0., cx],
                              [0., fy, cy],
                              [0., 0., 1.0]]).astype(np.float32)
        h = camera.height
        w = camera.width
        
        images_meta = read_extrinsics_binary(colmap_image_file_path)
        
        color_images_list = []
        image_paths_list = []
        c2ws_list = []
        intrinsics_list = []
        img_id_list = []

        i = 0
        for img_id, img_meta in images_meta.items():
            frame_name = img_meta.name
            frame_path = os.path.join(image_path, frame_name)

            q = img_meta.qvec
            t = img_meta.tvec
            r = qvec2rotmat(q)
            rt = np.eye(4)
            rt[:3,:3] = r
            rt[:3, 3] = t
            c2w = np.linalg.inv(rt).astype(np.float32)
            rgb = np.array(Image.open(frame_path))  # h, w, 3

            c2ws_list.append(c2w)
            intrinsics_list.append(intrinsic)
            image_paths_list.append(frame_path)
            color_images_list.append(rgb)
            img_id_list.append(img_id)

        # run metric3dv2
        depth_maps_list, normal_maps_list = extract_mono_geo_demo(color_images_list, intrinsics_list)

        cam_intrinsics, images_metas, points3d = read_model(os.path.join(data_path, "sparse"), ext=".bin")
        pts_indices = np.array([points3d[key].id for key in points3d])
        pts_xyzs = np.array([points3d[key].xyz for key in points3d])
        points3d_ordered = np.zeros([pts_indices.max()+1, 3])
        points3d_ordered[pts_indices] = pts_xyzs
        depth_maps_list_new = []
        for img_id, monodepth in zip(img_id_list, depth_maps_list):
            invmonodepthmap = 1 / monodepth
            res = get_scales(img_id, cam_intrinsics, images_metas, points3d_ordered, invmonodepthmap)
            scale = res['scale']
            offset = res['offset']
            if scale > 0:
                invmonodepthmap = invmonodepthmap * scale + offset
                monodepth = 1 / invmonodepthmap
            depth_maps_list_new.append(monodepth)

        data = {
            'color': color_images_list,
            'depth': depth_maps_list_new,
            'normal': normal_maps_list,
            'image_paths': image_paths_list,
            'extrinsics': c2ws_list,  # c2w
            'intrinsics': intrinsics_list,
            'out_path': out_path
        }
        torch.save(data, precomputed_data_path)

    # load conf
    base_conf = ConfigFactory.parse_file('planarsplat/confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(args.conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)
    conf.put('train.exps_folder_name', out_path)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.img_res', img_res)
    conf.put('dataset.pre_align', True)
    conf.put('dataset.voxel_length', 0.1)
    conf.put('dataset.sdf_trunc', 0.2)
    conf.put('plane_model.init_plane_num', 3000)

    planar_rec = run_planarSplatting(data=data, conf=conf)