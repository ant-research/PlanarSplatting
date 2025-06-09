import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_metric3d import extract_mono_geo_demo
from utils_demo.run_vggt import run_vggt
from utils_demo.misc import is_video_file, save_frames_from_video
from utils_demo.run_planarSplatting import run_planarSplatting



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", type=str, default='examples/living_room/images', help='path of input data')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/demo', help='path of output dir')
    parser.add_argument("-s", "--frame_step", type=int, default=10, help='sampling step of video frames')
    parser.add_argument("--depth_conf", type=float, default=2.0, help='depth confidence threshold of vggt')
    parser.add_argument("--conf_path", type=str, default='utils_demo/demo.conf', help='path of configure file')
    parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    
    image_path = None
    if os.path.isdir(data_path):
        image_path = data_path
    else:
        if is_video_file(data_path):
            absolute_video_path = os.path.abspath(data_path)
            current_dir = os.path.dirname(absolute_video_path)
            image_path = os.path.join(current_dir, 'images')
            save_frames_from_video(data_path, image_path, args.frame_step)
        else:
            raise ValueError(f'The input file {data_path} is not a video file.')
    assert image_path is not None, f"Can not find images or videos from {data_path}."

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    precomputed_data_path = os.path.join(out_path, 'data.pth')
    use_precomputed_data = args.use_precomputed_data
    
    if use_precomputed_data and os.path.exists(precomputed_data_path):
        data = torch.load(precomputed_data_path)
        print(f"loading precomputed data from {precomputed_data_path}")
    else:
        # run vggt
        data = run_vggt(image_path, out_path, depth_conf_thresh=args.depth_conf)

        # run metric3dv2
        _, normal_maps_list = extract_mono_geo_demo(data['color'], data['intrinsics'])
        data['normal'] = normal_maps_list
        torch.save(data, precomputed_data_path)

    # run planarSplatting
    '''
        data = {
            'color': [...],
            'depth': [...],
            'normal': [...],
            'image_paths': [...],
            'extrinsics': [...],  # c2w
            'intrinsics': [...],
        }
    '''
    # load conf
    base_conf = ConfigFactory.parse_file('planarsplat/confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(args.conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)
    conf.put('train.exps_folder_name', out_path)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.img_res', img_res)

    planar_rec = run_planarSplatting(data=data, conf=conf)


    