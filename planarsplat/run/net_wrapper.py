import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc_util import get_class
from utils import model_util
from loguru import logger


class PlanarRecWrapper(nn.Module):
    def __init__(self, conf, plane_plots_dir: str):
        super().__init__()
        self.conf = conf
        self.plane_model_conf = self.conf.get_config('plane_model')
        self.planarSplat = get_class(self.plane_model_conf.get_string('plane_model_class'))(conf, plot_dir=plane_plots_dir)

        self.init_type = self.plane_model_conf.get_string('init_type', default='mesh')
        # import pdb; pdb.set_trace()
        if self.init_type == 'mesh':
            try:
                logger.info('using mesh initialization......')
                mesh_path = self.conf.get_string('dataset.mesh_path')
                assert os.path.exists(mesh_path), f'mesh path: {mesh_path} does not exist'
                self.planarSplat.initialize_from_mesh(mesh_path)
            except:
                logger.info('using sphere initialization......')
                self.planarSplat.initialize_from_sphere()
        else:
            logger.info('using sphere initialization......')
            self.planarSplat.initialize_from_sphere()

        self.plane_vis_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")
        self.radii_grad_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")
        self.radii_grad_accum = torch.zeros((self.planarSplat.get_plane_num(), 4), device="cuda")
        self.split_thres = self.plane_model_conf.get_float('split_thres')
        self.radii_dir_type = self.plane_model_conf.get_string('radii_dir_type')

    def build_optimizer_and_LRscheduler(self):
        plane_model_conf = self.conf.get_config('plane_model')
        opt_dict = [
            {'params': [self.planarSplat._plane_center], 'lr': plane_model_conf.lr_center, "name": "plane_center", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_radii_xy_p], 'lr': plane_model_conf.lr_radii, "name": "plane_radii_xy_p", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_radii_xy_n], 'lr': plane_model_conf.lr_radii, "name": "plane_radii_xy_n", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_rot_q_normal_wxy], 'lr': plane_model_conf.lr_rot_normal, "name": "plane_rot_q_normal_wxy", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_rot_q_xyAxis_z], 'lr': plane_model_conf.lr_rot_xy, "name": "plane_rot_q_xyAxis_z", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_rot_q_xyAxis_w], 'lr': plane_model_conf.lr_rot_xy, "name": "plane_rot_q_xyAxis_w", "weight_decay": 0.},
        ]
        self.optimizer = torch.optim.Adam(opt_dict, betas=(0.9, 0.99), eps=1e-15)
        
    def update_learning_rate(self, ite):
        lr_dict = {}
        return lr_dict

    def prune_optimizer(self, valid_mask):
        valid_mask = valid_mask.squeeze()
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        assert len(optimizable_tensors) > 0
        return optimizable_tensors
    
    def prune_core(self, invalid_mask):
        invalid_mask = invalid_mask.squeeze()
        valid_plane_mask = ~invalid_mask
        opt_tensors = self.prune_optimizer(valid_plane_mask)

        self.planarSplat._plane_center = opt_tensors["plane_center"]
        self.planarSplat._plane_radii_xy_p = opt_tensors['plane_radii_xy_p']
        self.planarSplat._plane_radii_xy_n = opt_tensors['plane_radii_xy_n']
        self.planarSplat._plane_rot_q_normal_wxy = opt_tensors['plane_rot_q_normal_wxy']
        self.planarSplat._plane_rot_q_xyAxis_z = opt_tensors['plane_rot_q_xyAxis_z']
        self.planarSplat._plane_rot_q_xyAxis_w = opt_tensors['plane_rot_q_xyAxis_w']

        self.plane_vis_denom = self.plane_vis_denom[valid_plane_mask]
        self.radii_grad_denom = self.radii_grad_denom[valid_plane_mask]
        self.radii_grad_accum = self.radii_grad_accum[valid_plane_mask]

        if self.planarSplat.rot_delta is not None:
            self.planarSplat.rot_delta = self.planarSplat.rot_delta[valid_plane_mask]

    def prune_invisible_plane(self,):
        prune_mask = self.plane_vis_denom == 0
        self.prune_core(prune_mask.detach())
        self.planarSplat.check_model()
        self.reset_plane_vis()
        torch.cuda.empty_cache()

    def prune_small_plane(self, min_radii=None):
        plane_radii = self.planarSplat.get_plane_radii()
        if min_radii is None:
            prune_mask = plane_radii.abs().max(-1)[0] <= self.planarSplat.radii_min_list[-1] * 1.25
        else:
            prune_mask = plane_radii.abs().max(-1)[0] <= min_radii
        self.prune_core(prune_mask.detach())
        self.planarSplat.check_model()
        torch.cuda.empty_cache()
    
    def prune_overlapped_plane(self):
        plane_normal, plane_offset, plane_center, plane_radii, plane_rot_q, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry()
        prune_mask = model_util.get_overlapped_mask(plane_center, plane_normal, plane_offset, plane_radii, plane_xAxis, plane_yAxis)
        self.prune_core(prune_mask.detach().cuda().squeeze())
        self.planarSplat.check_model()
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z, new_rot_delta=[]):
        d = {"plane_center": new_plane_center,
            "plane_radii_xy_p": new_plane_radii_xy_p,
            "plane_radii_xy_n": new_plane_radii_xy_n,
            "plane_rot_q_normal_wxy": new_plane_rot_q_normal_wxy,
            "plane_rot_q_xyAxis_w" : new_plane_rot_q_xyAxis_w,
            "plane_rot_q_xyAxis_z" : new_plane_rot_q_xyAxis_z,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self.planarSplat._plane_center = optimizable_tensors["plane_center"]
        self.planarSplat._plane_radii_xy_p = optimizable_tensors["plane_radii_xy_p"]
        self.planarSplat._plane_radii_xy_n = optimizable_tensors["plane_radii_xy_n"]
        self.planarSplat._plane_rot_q_normal_wxy = optimizable_tensors["plane_rot_q_normal_wxy"]
        self.planarSplat._plane_rot_q_xyAxis_w = optimizable_tensors["plane_rot_q_xyAxis_w"]
        self.planarSplat._plane_rot_q_xyAxis_z = optimizable_tensors["plane_rot_q_xyAxis_z"]

        # reset all accum stats after densification
        self.reset_grad_stats()

        if self.planarSplat.rot_delta is not None and len(new_rot_delta) > 0:
            self.planarSplat.rot_delta = torch.cat([self.planarSplat.rot_delta, new_rot_delta], dim=0)

    def split_planes_via_radii_grad(self, radii_ratio=1.5):
        grads_radii = self.radii_grad_accum / self.radii_grad_denom
        grads_radii[grads_radii.isnan()] = 0.0
        assert self.planarSplat.get_plane_num() == grads_radii.shape[0]
        _, _, plane_center, plane_radii, _, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry()
        plane_radii_xy_p = plane_radii[:, :2]
        plane_radii_xy_n = plane_radii[:, 2:]
        plane_rot_q_normal_wxy = self.planarSplat.get_plane_rot_q_normal_wxy
        plane_rot_q_xyAxis_w = self.planarSplat.get_plane_rot_q_xyAxis_w
        plane_rot_q_xyAxis_z = self.planarSplat.get_plane_rot_q_xyAxis_z

        # Extract planes that satisfy the gradient condition
        x_split_mask, y_split_mask = model_util.get_split_mask_via_radii_grad(
            grads_radii, plane_radii_xy_p, plane_radii_xy_n, radii_ratio, self.planarSplat.radii_min_list[-1], self.split_thres)
        selected_mask_1 = torch.logical_and(y_split_mask, ~x_split_mask)
        selected_mask_2 = torch.logical_and(x_split_mask, ~y_split_mask)
        selected_mask_3 = torch.logical_and(x_split_mask, y_split_mask)
        selected_mask = selected_mask_1 | selected_mask_2 | selected_mask_3

        new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q_normal_wxy,new_plane_rot_q_xyAxis_w,new_plane_rot_q_xyAxis_z,new_rot_delta = model_util.split_planes_via_mask(
            selected_mask_1, selected_mask_2, selected_mask_3, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z, self.planarSplat.rot_delta)

        if len(new_plane_center) > 0:
            if self.planarSplat.rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)
            self.densification_postfix(new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z, new_rot_delta)
        
        self.planarSplat.check_model()
        self.reset_grad_stats()
        self.reset_plane_vis()
        if len(new_plane_center) > 0:
            prune_mask = torch.cat((selected_mask, torch.zeros(new_plane_center.shape[0], device="cuda", dtype=bool)))
            self.prune_core(prune_mask)
        torch.cuda.empty_cache()

    def split_all_plane(self,):
        self.regularize_plane_shape()
        _, _, plane_center, plane_radii, _, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry()
        plane_num = self.planarSplat.get_plane_num()
        plane_radii_xy_p = plane_radii[:, :2]
        plane_radii_xy_n = plane_radii[:, 2:]
        plane_rot_q_normal_wxy = self.planarSplat.get_plane_rot_q_normal_wxy
        plane_rot_q_xyAxis_w = self.planarSplat.get_plane_rot_q_xyAxis_w
        plane_rot_q_xyAxis_z = self.planarSplat.get_plane_rot_q_xyAxis_z

        x_split_mask = torch.ones(plane_num).cuda() > 0
        y_split_mask = torch.ones(plane_num).cuda() > 0
        selected_mask_1 = torch.logical_and(y_split_mask, ~x_split_mask)
        selected_mask_2 = torch.logical_and(x_split_mask, ~y_split_mask)
        selected_mask_3 = torch.logical_and(x_split_mask, y_split_mask)
        selected_mask = selected_mask_1 | selected_mask_2 | selected_mask_3

        new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q_normal_wxy,new_plane_rot_q_xyAxis_w,new_plane_rot_q_xyAxis_z,new_rot_delta = model_util.split_planes_via_mask(
            selected_mask_1, selected_mask_2, selected_mask_3, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z, self.planarSplat.rot_delta)

        if len(new_plane_center) > 0:
            if self.planarSplat.rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)
            self.densification_postfix(new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z, new_rot_delta)
        
        self.planarSplat.check_model()
        self.reset_grad_stats()
        self.reset_plane_vis()
        if len(new_plane_center) > 0:
            prune_mask = torch.cat((selected_mask, torch.zeros(new_plane_center.shape[0], device="cuda", dtype=bool)))
            self.prune_core(prune_mask)
        self.prune_small_plane()
        torch.cuda.empty_cache()

    def split_large_plane(self, radii_threshold=0.1):
        self.regularize_plane_shape()
        _, _, plane_center, plane_radii, _, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry()
        plane_num = self.planarSplat.get_plane_num()
        plane_radii_xy_p = plane_radii[:, :2]
        plane_radii_xy_n = plane_radii[:, 2:]
        plane_rot_q_normal_wxy = self.planarSplat.get_plane_rot_q_normal_wxy
        plane_rot_q_xyAxis_w = self.planarSplat.get_plane_rot_q_xyAxis_w
        plane_rot_q_xyAxis_z = self.planarSplat.get_plane_rot_q_xyAxis_z

        x_split_mask = plane_radii[:, 0] > radii_threshold
        y_split_mask = plane_radii[:, 1] > radii_threshold
        selected_mask_1 = torch.logical_and(y_split_mask, ~x_split_mask)
        selected_mask_2 = torch.logical_and(x_split_mask, ~y_split_mask)
        selected_mask_3 = torch.logical_and(x_split_mask, y_split_mask)
        selected_mask = selected_mask_1 | selected_mask_2 | selected_mask_3

        new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q_normal_wxy,new_plane_rot_q_xyAxis_w,new_plane_rot_q_xyAxis_z,new_rot_delta = model_util.split_planes_via_mask(
            selected_mask_1, selected_mask_2, selected_mask_3, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z, self.planarSplat.rot_delta)

        if len(new_plane_center) > 0:
            if self.planarSplat.rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)
            self.densification_postfix(new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z, new_rot_delta)
        
        self.planarSplat.check_model()
        self.reset_grad_stats()
        self.reset_plane_vis()
        if len(new_plane_center) > 0:
            prune_mask = torch.cat((selected_mask, torch.zeros(new_plane_center.shape[0], device="cuda", dtype=bool)))
            self.prune_core(prune_mask)
        self.prune_small_plane()
        torch.cuda.empty_cache()

    def split_plane(self,):
        self.regularize_plane_shape()
        self.split_planes_via_radii_grad()
        self.prune_small_plane()
        self.reset_plane_vis()
        self.reset_grad_stats()
        torch.cuda.empty_cache()

    def reset_plane_vis(self):
        self.plane_vis_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")

    def update_plane_vis(self, vis_mask=None):
        if vis_mask is None:
            vis_mask = self.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0

        assert vis_mask.shape[0] == self.plane_vis_denom.shape[0]
        self.plane_vis_denom[vis_mask] += 1
   
    def reset_grad_stats(self):
        self.radii_grad_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")
        self.radii_grad_accum = torch.zeros((self.planarSplat.get_plane_num(), 4), device="cuda")

    def update_grad_stats(self, vis_mask=None):
        if self.planarSplat._plane_radii_xy_p.grad is not None:
            if vis_mask is None:
                if self.planarSplat._plane_center.grad is not None:
                    vis_mask = self.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0
                else:
                    vis_mask = torch.ones(self.planarSplat.get_plane_num()).cuda() > 0
            self.radii_grad_denom = self.radii_grad_denom + vis_mask.float().view(-1, 1)
            if self.radii_dir_type == 'double':
                self.radii_grad_accum = self.radii_grad_accum + torch.cat([self.planarSplat._plane_radii_xy_p.grad, self.planarSplat._plane_radii_xy_n.grad], dim=-1).abs().detach() * vis_mask.float().view(-1, 1)
            elif self.radii_dir_type == 'single':
                self.radii_grad_accum[vis_mask] += torch.cat([self.planarSplat._plane_radii_xy_p.grad, self.planarSplat._plane_radii_xy_p.grad], dim=-1).abs().detach()[vis_mask]
            else:
                raise NotImplementedError
            
    def regularize_plane_shape(self, empty_cache=False):
        with torch.no_grad():
            plane_radii, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry_for_regularize()
            plane_center_x_offset = (plane_radii[:, 0] - plane_radii[:, 2]).unsqueeze(-1) / 2.
            plane_center_y_offset = (plane_radii[:, 1] - plane_radii[:, 3]).unsqueeze(-1) / 2.
            plane_center_offset_new = plane_center_x_offset * plane_xAxis + plane_center_y_offset * plane_yAxis
            plane_radii_x_new = (plane_radii[:, 0] + plane_radii[:, 2]) / 2.
            plane_radii_y_new = (plane_radii[:, 1] + plane_radii[:, 3]) / 2.
            plane_radii_new = torch.stack([plane_radii_x_new, plane_radii_y_new], dim=-1)
            plane_center_new = self.planarSplat._plane_center + plane_center_offset_new

        # reset plane parameters
        opt_tensors = self.replace_tensor_to_optimizer(plane_center_new.detach(), 'plane_center')
        self.planarSplat._plane_center = opt_tensors['plane_center']
        opt_tensors = self.replace_tensor_to_optimizer(plane_radii_new.detach(), 'plane_radii_xy_p')
        self.planarSplat._plane_radii_xy_p = opt_tensors['plane_radii_xy_p']
        opt_tensors = self.replace_tensor_to_optimizer(plane_radii_new.detach(), 'plane_radii_xy_n')
        self.planarSplat._plane_radii_xy_n = opt_tensors['plane_radii_xy_n']
        if empty_cache:
            torch.cuda.empty_cache()

    def forward(self):
        pass
    
    