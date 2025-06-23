import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch
from random import randint
import math
from loguru import logger
import open3d as o3d
from .net_wrapper import PlanarRecWrapper

from utils.misc_util import setup_logging, get_train_param, save_config_files, prepare_folders, get_class
from utils.trainer_util import resume_model, calculate_plane_depth, plot_plane_img, save_checkpoints
from utils.mesh_util import get_coarse_mesh
from utils.merge_util import merge_plane
from utils.loss_util import normal_loss, metric_depth_loss

import rerun as rr

class PlanarSplatTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        self.conf = kwargs['conf']
        self.expname, self.scan_id, self.timestamp, is_continue = get_train_param(kwargs, self.conf)
        self.expdir, self.plane_plots_dir, self.checkpoints_path, self.model_subdir = prepare_folders(kwargs, self.expname, self.timestamp)
        setup_logging(os.path.join(self.expdir, 'train.log'))
        
        logger.info('Shell command : {0}'.format(' '.join(sys.argv)))
        save_config_files(self.expdir, self.conf)

        # =======================================  loading dataset
        logger.info('Loading data...')
        if 'data' in kwargs:
            self.dataset = get_class(self.conf.get_string('train.dataset_class'))(kwargs['data'], **self.conf.get_config('dataset'))
        else:
            self.dataset = get_class(self.conf.get_string('train.dataset_class'))(**self.conf.get_config('dataset'))
        self.ds_len = self.dataset.n_images
        self.H = self.conf.dataset.img_res[0]
        self.W = self.conf.dataset.img_res[1]
        logger.info('Data loaded. Frame number = {0}'.format(self.ds_len))

        # =======================================  build plane model
        self.plane_model_conf = self.conf.get_config('plane_model')
        self.conf['dataset']['mesh_path'] = self.dataset.mono_mesh_dest

        net = PlanarRecWrapper(self.conf, self.plane_plots_dir)
        self.net = net.cuda()
        self.resumed = False
        self.start_iter = resume_model(self) if is_continue else 0
        self.iter_step = self.start_iter
        self.net.build_optimizer_and_LRscheduler()

        # ======================================= plot settings
        self.do_vis = kwargs['do_vis']
        self.plot_freq = self.conf.get_int('train.plot_freq')        
        
        # ======================================= loss settings
        loss_plane_conf = self.conf.get_config('plane_model.plane_loss')
        self.weight_plane_normal = loss_plane_conf.get_float('weight_mono_normal')
        self.weight_plane_depth = loss_plane_conf.get_float('weight_mono_depth')

        # ======================================= training settings
        self.max_total_iters = self.conf.get_int('train.max_total_iters')
        self.process_plane_freq_ite = self.conf.get_int('train.process_plane_freq_ite')
        self.coarse_stage_ite = self.conf.get_int('train.coarse_stage_ite')
        self.split_start_ite = self.conf.get_int('train.split_start_ite')
        self.check_vis_freq_ite = self.conf.get_int('train.check_plane_vis_freq_ite')
        self.data_order = self.conf.get_string('train.data_order')

    def run(self):
        self.train()
        self.merger()
    
    def train(self):
        logger.info("Training...")
        if self.start_iter >= self.max_total_iters:
            return
        weight_decay_list = []
        for i in tqdm(range(self.max_total_iters+1), desc="generating sampling idx list..."):
            weight_decay_list.append(max(math.exp(-i / self.max_total_iters), 0.1))
        logger.info('Start training at {:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()))
        self.net.train()
        if self.iter_step == 0:
            self.check_plane_visibility_cuda()  

        view_info_list = None
        progress_bar = tqdm(range(self.start_iter, self.max_total_iters+1), desc="Training progress")
        calculate_plane_depth(self)
        for iter in range(self.start_iter, self.max_total_iters + 1):
            self.iter_step = iter
            # ======================================= process planes
            if iter > self.coarse_stage_ite and iter % self.process_plane_freq_ite==0:  
                self.net.regularize_plane_shape()
                self.net.prune_small_plane()
                if iter > self.split_start_ite and iter <= self.max_total_iters - 1000:
                    logger.info('splitting...')
                    ori_num = self.net.planarSplat.get_plane_num()
                    self.net.split_plane()
                    new_num = self.net.planarSplat.get_plane_num()
                    logger.info(f'plane num: {ori_num} ---> {new_num}')
            # ======================================= get view info
            if not view_info_list:
                view_info_list = self.dataset.view_info_list.copy()
            if self.data_order == 'rand':
                view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            else:
                view_info = view_info_list.pop(0)
            raster_cam_w2c = view_info.raster_cam_w2c
            # ======================================= zero grad
            self.net.optimizer.zero_grad()
            #  ======================================= plane forward
            allmap = self.net.planarSplat(view_info,iter)
            # ------------ get rendered maps
            depth = allmap[0:1].squeeze().view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
            # ------------ get aux maps
            vis_weight = allmap[1:2].squeeze().view(-1)
            valid_ray_mask = vis_weight > 0.00001
            valid_normal_mask = view_info.mono_normal_global.abs().sum(dim=-1) > 0
            valid_depth_mask = view_info.mono_depth.abs() > 0
            valid_ray_mask = valid_ray_mask & valid_depth_mask & valid_normal_mask

            # ======================================= calculate losses
            loss_final = 0.
            decay = weight_decay_list[iter]
            # ------------ calculate plane loss
            loss_plane_normal_l1, loss_plane_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
            loss_plane_depth = metric_depth_loss(depth, view_info.mono_depth, valid_ray_mask, max_depth=10.0)
            loss_plane = (loss_plane_depth * 1.0) * self.weight_plane_depth \
                        + (loss_plane_normal_l1 + loss_plane_normal_cos) * self.weight_plane_normal
            loss_final += loss_plane * decay

            # ======================================= backward & update plane denom & update learning rate
            loss_final.backward()
            self.net.optimizer.step()
            self.net.update_grad_stats()
            self.net.regularize_plane_shape(False)
            image_index = view_info.index
            self.dataset.view_info_list[image_index].plane_depth = depth.detach().clone()

            with torch.no_grad():
                # Progress bar
                plane_num = self.net.planarSplat.get_plane_num()
                if iter % 10 == 0:
                    loss_dict = {
                        "Planes": f"{plane_num}",
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iter == self.max_total_iters:
                    progress_bar.close()
            
            # ======================================= plot model outputs
            if self.do_vis and iter % self.plot_freq == 0:
                self.net.regularize_plane_shape()
                self.net.eval()
                self.net.planarSplat.draw_plane(epoch=iter)
                plot_plane_img(self)
                self.net.train()
            
            if iter > 0 and iter % self.check_vis_freq_ite == 0:
                self.check_plane_visibility_cuda()
        
        
        self.check_plane_visibility_cuda()
        save_checkpoints(self, iter=self.iter_step, only_latest=False)

    def merger(self, save_mesh=True):
        logger.info("Merging 3D planar primitives...")
        output_dir = self.conf.get_string('train.rec_folder_name', default='')
        if len(output_dir) == 0:
            output_dir = self.expdir
        self.net.eval()
        save_root = os.path.join(output_dir, f'{self.scan_id}')
        os.makedirs(save_root, exist_ok=True)

        ## prune planes whose maximum radii lower than the threshold
        self.net.prune_small_plane(min_radii=0.02 * self.net.planarSplat.pose_cfg.scale)
        logger.info("number of 3D planar primitives = %d"%(self.net.planarSplat.get_plane_num()))

        coarse_mesh = get_coarse_mesh(
            self.net, 
            self.dataset.view_info_list.copy(), 
            self.H, 
            self.W, 
            voxel_length=0.02, 
            sdf_trunc=0.08)
        
        merge_config_coarse = self.conf.get_config('merge_coarse', default=None)
        merge_config_fine = self.conf.get_config('merge_fine', default=None)
        if merge_config_coarse is not None:
            logger.info(f'mergeing (coarse)...')
            planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                self.net, 
                coarse_mesh, 
                plane_ins_id=None,
                **merge_config_coarse)
            if merge_config_fine is not None:
                logger.info(f'mergeing (fine)...')
                planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                    self.net, 
                    coarse_mesh, 
                    plane_ins_id=plane_ins_id_new,
                    **merge_config_fine)
        else:
            raise ValueError("No merge configuration found!")
        
        if save_mesh:
            save_path = os.path.join(save_root, f"{self.scan_id}_planar_mesh.ply")
            logger.info(f'saving final planar mesh to {save_path}')
            o3d.io.write_triangle_mesh(
                        save_path, 
                        planarSplat_eval_mesh)
        return planarSplat_eval_mesh

    def check_plane_visibility_cuda(self):   
        self.net.regularize_plane_shape(False)     
        logger.info('checking plane visibility...')
        self.net.eval()
        self.net.reset_plane_vis()
        view_info_list = self.dataset.view_info_list.copy()
        for iter in tqdm(range(self.ds_len)):
            # ========================= get view info
            view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            raster_cam_w2c = view_info.raster_cam_w2c
            # ----------- plane forward
            allmap = self.net.planarSplat(view_info, self.iter_step)
            # get rendered maps
            depth = allmap[0:1].view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
            # get aux maps
            vis_weight = allmap[1:2].view(-1)
            valid_ray_mask = vis_weight > 0.00001

            loss_final = 0.
            # ======================================= calculate plane losses
            loss_mono_depth = metric_depth_loss(depth, view_info.mono_depth, valid_ray_mask)
            loss_normal_l1, loss_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
            loss_final += loss_mono_depth + loss_normal_cos + loss_normal_l1

            loss_final.backward() 
            # update plane visibility
            self.net.update_plane_vis() 
            self.net.optimizer.zero_grad()

        self.net.optimizer.zero_grad()
        self.net.train()
        self.net.prune_invisible_plane()
        self.net.planarSplat.draw_plane(epoch=self.iter_step)
    
    
    
    