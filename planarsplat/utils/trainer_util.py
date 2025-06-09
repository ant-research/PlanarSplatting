import os
import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from random import randint

def plot_plane_img(trainer, plot_img_idx: int=0, prefix: str='', pcd_on: bool=False):
    trainer.net.regularize_plane_shape(False)
    view_info_list = trainer.dataset.view_info_list.copy()
    if plot_img_idx < 0:
        plot_img_idx = randint(0, len(view_info_list)-1)
    view_info = view_info_list[plot_img_idx]
    raster_cam_w2c = view_info.raster_cam_w2c
    cam_loc = view_info.cam_loc
    
    rendered_rgb, allmap = trainer.net.planarSplat(view_info, trainer.iter_step, return_rgb=True)

    # get rendered maps
    rendered_depth = allmap[0:1].view(-1)
    normal_local_ = allmap[2:5]
    
    rendered_normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
    rendered_normal_np = (F.normalize(rendered_normal_global, dim=-1).reshape(trainer.H, trainer.W, 3)).detach().cpu().numpy()
    rendered_normal_color = ((rendered_normal_np + 1.0) / 2.0 * 255).astype(np.uint8)
    rendered_rgb_np = (rendered_rgb.permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)
    rendered_depth_np = rendered_depth.reshape(trainer.H, trainer.W).detach().cpu().numpy()

    if pcd_on:
        depth_scale = view_info.depth_scale
        ray_dirs = view_info.ray_dirs
        plane_surf = cam_loc.view(1, 3) + rendered_depth.view(-1, 1) / depth_scale.view(-1, 1) * ray_dirs.view(-1, 3)
        plane_surf_np = plane_surf.detach().cpu().numpy()
        scene_scale = trainer.conf.pose.scale
        scene_offset = np.asarray(trainer.conf.pose.offset)
        plane_surf_np /= scene_scale
        plane_surf_np += scene_offset

    # get gt
    gt_rgb_np = (view_info.rgb.reshape(trainer.H, trainer.W, 3) * 255).cpu().numpy().astype(np.uint8)
    gt_normal_np = (F.normalize(view_info.mono_normal_global, dim=-1).reshape(trainer.H, trainer.W, 3)).cpu().numpy()
    gt_normal_color = ((gt_normal_np + 1.0) / 2.0 * 255).astype(np.uint8)
    gt_depth_np = view_info.mono_depth.reshape(trainer.H, trainer.W).cpu().numpy()

    n_r = 1
    n_c = 5
    plt.figure(figsize=(8, 2.5))  # width, height
    plt.subplot(n_r, n_c, 1)
    plt.title("rgb gt")
    plt.imshow(gt_rgb_np)
    plt.axis('off')

    plt.subplot(n_r, n_c, 2)
    plt.title("mono normal")
    plt.imshow(gt_normal_color)
    plt.axis('off')

    plt.subplot(n_r, n_c, 3)
    plt.title("rendered normal")
    plt.imshow(rendered_normal_color)
    plt.axis('off')

    plt.subplot(n_r, n_c, 4)
    plt.title("mono depth")
    plt.imshow(gt_depth_np)
    plt.axis('off')

    plt.subplot(n_r, n_c, 5)
    plt.title("rendered depth")
    plt.imshow(rendered_depth_np)
    plt.axis('off')

    plt.tight_layout()

    root_dir = trainer.plane_plots_dir
    os.makedirs(root_dir, exist_ok=True)
    save_dir = os.path.join(root_dir, '%svis_%d_%d_cuda.jpg'%(prefix, trainer.iter_step, plot_img_idx))
    plt.savefig(save_dir, pad_inches=0)
    logger.info(f"saving to {save_dir}")

    save_dir = os.path.join(root_dir, 'debug_vis_%d_cuda.jpg'%(plot_img_idx))
    plt.savefig(save_dir, pad_inches=0)
    logger.info(f"saving to {save_dir}")

    plt.close()

    # plt.imsave(os.path.join(root_dir, 'tmp_cuda.jpg'), rendered_depth_np, cmap='viridis')

    if pcd_on:
        points_plane_o3d = o3d.geometry.PointCloud()
        points_plane_o3d.points = o3d.utility.Vector3dVector(plane_surf_np)
        save_dir = os.path.join(root_dir, 'debug_vis_%d_plane_surf.ply'%(plot_img_idx))
        logger.info(f'saving to {save_dir}')
        o3d.io.write_point_cloud(save_dir, points_plane_o3d)

def calculate_plane_depth(trainer):   
    trainer.net.regularize_plane_shape(False)     
    trainer.net.eval()
    view_info_list = trainer.dataset.view_info_list.copy()
    with torch.no_grad():
        for iter in range(trainer.ds_len):
            # ========================= get view info
            view_info = view_info_list[iter]
            # ----------- plane forward
            with torch.no_grad():
                allmap = trainer.net.planarSplat(view_info, trainer.iter_step)
            # get rendered maps
            depth = allmap[0:1].view(-1)
            trainer.dataset.view_info_list[iter].plane_depth = depth.detach()
    trainer.net.train()

def save_checkpoints(trainer, iter, only_latest=False):
    torch.save(
            {"iter": iter, "model_state_dict": trainer.net.state_dict()},
            os.path.join(trainer.checkpoints_path, trainer.model_subdir, "latest.pth"))
    if not only_latest:
        torch.save(
            {"iter": iter, "model_state_dict": trainer.net.state_dict()},
            os.path.join(trainer.checkpoints_path, trainer.model_subdir, str(iter) + ".pth"))

def resume_model(trainer, ckpt_name='latest'):
        logger.info('resuming...')
        old_checkpnts_dir = trainer.checkpoints_path
        if not os.path.exists(os.path.join(old_checkpnts_dir, trainer.model_subdir, ckpt_name + ".pth")):
            ckpts = os.listdir(os.path.join(old_checkpnts_dir, trainer.model_subdir))
            ckpts.sort()
            ckpt_name = ckpts[-1].split('.')[0]
        saved_model_state = torch.load(os.path.join(old_checkpnts_dir, trainer.model_subdir, ckpt_name + ".pth"))
        logger.info(f'loading model from {os.path.join(old_checkpnts_dir, trainer.model_subdir, ckpt_name + ".pth")}')
        plane_num = saved_model_state["model_state_dict"]['planarSplat._plane_center'].shape[0]
        trainer.net.planarSplat.initialize_as_zero(plane_num)
        trainer.net.build_optimizer_and_LRscheduler()
        trainer.net.reset_plane_vis()
        trainer.net.reset_grad_stats()
        trainer.resumed = True
        trainer.net.load_state_dict(saved_model_state["model_state_dict"])
        latest_iter = saved_model_state["iter"]
        return latest_iter + 1