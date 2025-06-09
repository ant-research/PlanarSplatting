import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import trimesh
from loguru import logger
from utils import model_util
from utils.plot_util import plot_rectangle_planes
from diff_rect_rasterization import RectRasterizationSettings, RectRasterizer 
from quaternion_utils._C import q2RCUDA, qMultCUDA


class PlanarSplat_Network(nn.Module):
    def __init__(self, cfg, plot_dir='',):
        super().__init__()
        # ==================================
        self.rot_delta = None
        self.bg = torch.tensor([0., 0., 0.]).cuda()
        # ================================== get config ======================
        self.plane_cfg = cfg.get_config('plane_model')
        self.pose_cfg = cfg.get_config('pose')
        self.data_cfg = cfg.get_config('dataset')
        self.plot_dir = plot_dir
        # ---------debug
        self.grads = None
        self.debug_on = self.plane_cfg.get_bool("debug_on", default=False)
        # data and scene setting
        self.H = self.data_cfg.img_res[0]
        self.W = self.data_cfg.img_res[1]
        self.scene_bounding_sphere = self.data_cfg.get_float('scene_bounding_sphere')
        # training setting
        self.max_total_iters = cfg.get_int('train.max_total_iters', default=5000)
        self.coarse_stage_ite = cfg.get_int('train.coarse_stage_ite')
        self.initialized = False
        # plane setting: plane splatting
        self.RBF_type = self.plane_cfg.get_string('RBF_type')
        self.RBF_weight_change_type = self.plane_cfg.get_string('RBF_weight_change_type')
        # plane setting: plane radii
        self.radii_dir_type = self.plane_cfg.get_string('radii_dir_type')
        logger.info(f"--------------------> radii type = {self.radii_dir_type}")
        self.init_radii = self.plane_cfg.get_float('radii_init')
        self.radii_max_list = self.plane_cfg.radii_max_list
        self.radii_min_list = self.plane_cfg.radii_min_list
        self.radii_milestone_list = self.plane_cfg.radii_milestone_list
        self.min_radii = 0.
        self.max_radii = 1000.

        # ================================== define plane parameters ======================
        # fixed plane parameters
        self.plane_normal_standard = torch.tensor([0., 0., 1.]).reshape(1, 3).cuda()
        self.plane_xAxis_standard = torch.tensor([1., 0., 0.]).reshape(1, 3).cuda()
        self.plane_yAxis_standard = torch.tensor([0., 1., 0.]).reshape(1, 3).cuda()
        self.plane_rot_q_normal_z = torch.Tensor(1, 1).fill_(0).cuda()
        self.plane_rot_q_xyAxis_xy = torch.Tensor(1, 2).fill_(0).cuda()
        self.plane_rot_q_xyAxis_xy_padded = self.plane_rot_q_xyAxis_xy.repeat(50000, 1)
        self.plane_rot_q_normal_z_padded = self.plane_rot_q_normal_z.repeat(50000, 1)
        # learnable plane parameters
        self._plane_center = torch.empty(0)
        self._plane_radii_xy_p = torch.empty(0)
        self._plane_radii_xy_n = torch.empty(0)
        self._plane_rot_q_normal_wxy = torch.empty(0)
        self._plane_rot_q_xyAxis_w = torch.empty(0)
        self._plane_rot_q_xyAxis_z = torch.empty(0)

    @property
    def get_plane_center(self):
        return self._plane_center
    
    @property
    def get_plane_radii_xy_p(self):
        return self._plane_radii_xy_p
    
    @property
    def get_plane_radii_xy_n(self):
        return self._plane_radii_xy_n
    
    @property
    def get_plane_rot_q_normal_wxy(self):
        return self._plane_rot_q_normal_wxy
    
    @property
    def get_plane_rot_q_xyAxis_w(self):
        return self._plane_rot_q_xyAxis_w
    
    @property
    def get_plane_rot_q_xyAxis_z(self):
        return self._plane_rot_q_xyAxis_z
    
    def initialize_from_sphere(self):
        logger.info("Initializing planes from sphere...")
        self.initialized = True
        plane_num = self.plane_cfg.get_int('init_plane_num')
        ratio = self.data_cfg.get_float('sphere_ratio', default=0.5)
        radius = self.scene_bounding_sphere * ratio
        init_centers, init_rot_q_normal, init_rot_q_xyAxis = model_util.get_plane_param_from_sphere(plane_num, radius)

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_centers.cuda().requires_grad_(True))
        self._plane_radii_xy_p = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
        self._plane_radii_xy_n = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis[:, 3:4].cuda(), requires_grad=True)

        # =========================  plane visualization  ======================
        self.draw_plane(epoch=-1, suffix='initial-sphere')

    def initialize_with_given_pts(self, pts, normals, radiis=None):
        self.initialized = True
        init_centers = pts
        plane_num = init_centers.shape[0]

        init_rot_q_normal = model_util.get_rotation_quaternion_of_normal(normals)
        init_rot_angle_xyAxis = torch.tensor([0.]).reshape(1, 1).repeat(normals.shape[0], 1)
        init_rot_q_xyAxis = model_util.get_rotation_quaternion_of_xyAxis(normals.shape[0], angle=init_rot_angle_xyAxis)

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_centers.cuda().requires_grad_(True))
        if radiis is None:
            self._plane_radii_xy_p = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
            self._plane_radii_xy_n = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
        else:
            self._plane_radii_xy_p = nn.Parameter(radiis[:, :2].cuda(), requires_grad=True)
            self._plane_radii_xy_n = nn.Parameter(radiis[:, 2:].cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis[:, 3:4].cuda(), requires_grad=True)

    def initialize_from_mesh(self, mesh_path):
        self.initialized = True
        plane_num = self.plane_cfg.get_int('init_plane_num')
        ratio = self.data_cfg.get_float('sphere_ratio', default=0.5)
        radius = self.scene_bounding_sphere * ratio
        _, _, init_rot_q_xyAxis = model_util.get_plane_param_from_sphere(plane_num, radius)

        mesh = trimesh.load_mesh(mesh_path)
        target_vertex_count = min(plane_num * 2, 5000)
        simplified_mesh = mesh.simplify_quadratic_decimation(target_vertex_count)
        vertices = simplified_mesh.vertices
        normals = simplified_mesh.vertex_normals
        faces = simplified_mesh.faces
        if faces.shape[0] < target_vertex_count:
            faces = np.concatenate((faces, faces[:target_vertex_count-faces.shape[0]]), axis=0)
        elif faces.shape[0] > target_vertex_count:
            faces = faces[:target_vertex_count]
        else:
            pass
        faces_v = vertices[faces.reshape(-1)].reshape(target_vertex_count, 3, 3)
        faces_n = normals[faces.reshape(-1)].reshape(target_vertex_count, 3, 3)
        plane_centers = (np.mean(faces_v, axis=1) - self.pose_cfg.offset) * self.pose_cfg.scale
        plane_normals = np.mean(faces_n, axis=1)
        plane_radii = np.mean(np.linalg.norm(plane_centers[:, None] -(faces_v- self.pose_cfg.offset) * self.pose_cfg.scale, axis=-1), axis=-1)

        sampled_idx = random.sample(range(0, target_vertex_count), plane_num)
        plane_centers = plane_centers[sampled_idx]
        plane_normals = plane_normals[sampled_idx]
        plane_radii = plane_radii[sampled_idx]

        init_centers_new = torch.from_numpy(plane_centers).float()
        init_normals_new = torch.from_numpy(plane_normals).float()
        init_rot_q_normal_new = model_util.get_rotation_quaternion_of_normal(init_normals_new).cuda()
        init_rot_q_xyAxis_new = init_rot_q_xyAxis
        init_radii = torch.from_numpy(plane_radii.reshape(-1, 1))
        init_radii = torch.cat([init_radii, init_radii], dim=-1).float() * 0.3

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_centers_new.cuda().requires_grad_(True))
        self._plane_radii_xy_p = nn.Parameter(init_radii.cuda(), requires_grad=True)
        self._plane_radii_xy_n = nn.Parameter(init_radii.cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal_new[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis_new[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis_new[:, 3:4].cuda(), requires_grad=True)

         # =========================  plane visualization  ======================
        self.draw_plane(epoch=-1, suffix='initial-mesh')
    
    def initialize_as_zero(self, plane_num):
        self.initialized = True
        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(torch.zeros(plane_num, 3).cuda().requires_grad_(True))
        self._plane_radii_xy_p = nn.Parameter(torch.zeros(plane_num, 2).cuda().requires_grad_(True))
        self._plane_radii_xy_n = nn.Parameter(torch.zeros(plane_num, 2).cuda().requires_grad_(True))
        self._plane_rot_q_normal_wxy = nn.Parameter(torch.zeros(plane_num, 3).cuda().requires_grad_(True))
        self._plane_rot_q_xyAxis_w = nn.Parameter(torch.zeros(plane_num, 1).cuda().requires_grad_(True))
        self._plane_rot_q_xyAxis_z = nn.Parameter(torch.zeros(plane_num, 1).cuda().requires_grad_(True))

    def check_model(self):
        assert self.get_plane_center.shape[0] == self.get_plane_radii_xy_p.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_radii_xy_n.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_rot_q_normal_wxy.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_rot_q_xyAxis_w.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_rot_q_xyAxis_z.shape[0]

    def get_plane_num(self):
        self.check_model()
        return self.get_plane_center.shape[0]
    
    def set_max_and_min_radii(self, ite):
        max_radii, min_radii = model_util.get_max_and_min_radii(ite, self.radii_max_list, self.radii_min_list, self.radii_milestone_list)
        self.max_radii = max_radii
        self.min_radii = min_radii
 
    def get_plane_rot_q(self):
        plane_num = self.get_plane_num()
        plane_rot_q_xyAxis = torch.cat([self._plane_rot_q_xyAxis_w, self.plane_rot_q_xyAxis_xy_padded[:plane_num], self._plane_rot_q_xyAxis_z], dim=-1)
        plane_rot_q_normal = torch.cat([self._plane_rot_q_normal_wxy, self.plane_rot_q_normal_z_padded[:plane_num]], dim=-1)
        plane_rot_q_xyAxis = F.normalize(plane_rot_q_xyAxis, dim=-1)
        plane_rot_q_normal = F.normalize(plane_rot_q_normal, dim=-1)
        return plane_rot_q_xyAxis, plane_rot_q_normal
    
    def get_plane_radii(self):
        if self.radii_dir_type == 'double':
            plane_radii = torch.cat([self._plane_radii_xy_p, self._plane_radii_xy_n], dim=-1)
        elif self.radii_dir_type == 'single':
            plane_radii = torch.cat([self._plane_radii_xy_p, self._plane_radii_xy_p], dim=-1)
        else:
            raise
        return plane_radii.clamp(min=self.min_radii, max=self.max_radii)

    def get_plane_geometry(self, ite=-1):
        fix_rot_normal = self.plane_cfg.fix_rot_normal
        fix_rot_xy = self.plane_cfg.fix_rot_xy
        fix_center = self.plane_cfg.fix_center
        fix_radii = True if (ite < self.coarse_stage_ite and not self.debug_on) else self.plane_cfg.fix_radii

        plane_num = self.get_plane_num()
        plane_rot_q_xyAxis, plane_rot_q_normal = self.get_plane_rot_q()
        plane_radii = self.get_plane_radii()
        
        if fix_rot_normal:
            plane_rot_q_normal = plane_rot_q_normal.detach()
        if fix_rot_xy:
            plane_rot_q_xyAxis = plane_rot_q_xyAxis.detach()
        if fix_radii:
            plane_radii = plane_radii.detach()
        
        plane_rot_q = model_util.quaternion_mult(plane_rot_q_normal, plane_rot_q_xyAxis)
        plane_rot_q = F.normalize(plane_rot_q, dim=-1)
        if self.rot_delta is not None:
            assert self.rot_delta.shape[0] == plane_rot_q.shape[0]
            plane_rot_q = model_util.quaternion_mult(self.rot_delta, plane_rot_q)
        plane_rot_matrix = model_util.quat_to_rot(plane_rot_q)

        plane_normal = torch.bmm(plane_rot_matrix, self.plane_normal_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        plane_xAxis = torch.bmm(plane_rot_matrix, self.plane_xAxis_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        plane_yAxis = torch.bmm(plane_rot_matrix, self.plane_yAxis_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)

        plane_center = self.get_plane_center
        if fix_center:
            plane_center = plane_center.detach()

        plane_offset = model_util.compute_offset(plane_center, plane_normal).reshape(-1, 1).float()

        return plane_normal, plane_offset, plane_center, plane_radii, plane_rot_q, plane_xAxis, plane_yAxis
    
    def get_plane_geometry_simple(self, ite=-1, in_fix_rot_n=False, in_fix_rot_xy=False, in_fix_radii=False, in_fix_center=False):
        fix_rot_normal = self.plane_cfg.fix_rot_normal or in_fix_rot_n
        fix_rot_xy = self.plane_cfg.fix_rot_xy or in_fix_rot_xy
        fix_center = self.plane_cfg.fix_center or in_fix_center
        fix_radii = True if (ite < self.coarse_stage_ite and not self.debug_on) else self.plane_cfg.fix_radii
        fix_radii = fix_radii or in_fix_radii

        plane_rot_q_xyAxis, plane_rot_q_normal = self.get_plane_rot_q()
        plane_radii = self.get_plane_radii()
        if ite < self.coarse_stage_ite:
            plane_radii = plane_radii.detach()
        
        if fix_rot_normal:
            plane_rot_q_normal = plane_rot_q_normal.detach()
        if fix_rot_xy:
            plane_rot_q_xyAxis = plane_rot_q_xyAxis.detach()
        if fix_radii:
            plane_radii = plane_radii.detach()
        
        plane_rot_q = model_util.quaternion_mult(plane_rot_q_normal, plane_rot_q_xyAxis)
        plane_rot_q = F.normalize(plane_rot_q, dim=-1)
        if self.rot_delta is not None:
            assert self.rot_delta.shape[0] == plane_rot_q.shape[0]
            plane_rot_q = model_util.quaternion_mult(self.rot_delta, plane_rot_q)

        plane_center = self._plane_center 
        if fix_center:
            plane_center = plane_center.detach()

        return plane_center, plane_radii, plane_rot_q

    def get_plane_geometry_for_regularize(self):
        plane_num = self.get_plane_num()
        plane_rot_q_xyAxis, plane_rot_q_normal = self.get_plane_rot_q()
        plane_radii = self.get_plane_radii()
        plane_rot_q = qMultCUDA(plane_rot_q_normal, plane_rot_q_xyAxis)

        plane_rot_q = F.normalize(plane_rot_q, dim=-1)
        if self.rot_delta is not None:
            raise ValueError
        plane_rot_matrix = q2RCUDA(plane_rot_q).permute(0, 2, 1)
        plane_xAxis = torch.bmm(plane_rot_matrix, self.plane_xAxis_standard.view(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        plane_yAxis = torch.bmm(plane_rot_matrix, self.plane_yAxis_standard.view(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        return plane_radii, plane_xAxis, plane_yAxis
    
    def get_splat_weight(self, ite=-1):
        if self.RBF_type == 'rectangle':
            if ite == -1 or self.RBF_weight_change_type == 'max':
                weight = 300.
            elif self.RBF_weight_change_type == 'increase':
                max_weight = 300.
                ratio = ite / (self.max_total_iters // 10)
                weight = min(math.exp(-(1 - ratio)) * 20, max_weight)
            elif self.RBF_weight_change_type == 'min':
                max_weight = 300.
                ratio = 0.
                weight = min(math.exp(-(1 - ratio)) * 20, max_weight)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return weight

    def forward(self, view_info, iter=-1, return_rgb=False, fix_rot_n=False, fix_rot_xy=False, fix_radii=False, fix_center=False):
        raster_cam_w2c = view_info.raster_cam_w2c
        raster_cam_fullproj = view_info.raster_cam_fullproj
        raster_cam_center = view_info.raster_cam_center
        tanfovx = view_info.tanfovx
        tanfovy = view_info.tanfovy
        raster_img_center = view_info.raster_img_center

        # ======================================= set up plane model
        self.set_max_and_min_radii(iter)
        plane_center, plane_radii, plane_rot_q = self.get_plane_geometry_simple(
                                                    ite=iter, 
                                                    in_fix_rot_n=fix_rot_n, 
                                                    in_fix_rot_xy=fix_rot_xy, 
                                                    in_fix_radii=fix_radii, 
                                                    in_fix_center=fix_center
                                                    )

        # ======================================= set up rasterization configuration
        splat_weight = self.get_splat_weight(ite=iter)
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(plane_center, dtype=plane_center.dtype, requires_grad=True, device="cuda")
        try:
            screenspace_points.retain_grad()
        except:
            pass
        raster_settings = RectRasterizationSettings(
                    image_height=self.H,
                    image_width=self.W,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg,
                    scale_modifier=1.0,
                    viewmatrix=raster_cam_w2c,
                    projmatrix=raster_cam_fullproj,
                    sh_degree=0,
                    campos=raster_cam_center,
                    prefiltered=False,
                    debug=False,
                    lambdaw=splat_weight * 5.0,
                    image_center=raster_img_center,
                    scales2=plane_radii[:, :2].detach()
        )

        # ======================================= plane model forward
        rasterizer = RectRasterizer(raster_settings=raster_settings)
        rgb, _, allmap = rasterizer(
                    means3D = plane_center,
                    means2D = screenspace_points,
                    shs = None,
                    colors_precomp = torch.rand_like(plane_center),
                    opacities = torch.ones_like(plane_center)[:, :1],
                    scales = plane_radii,
                    rotations = plane_rot_q,
                    cov3D_precomp = None
        )
        if return_rgb:
            return rgb, allmap
        else:
            return allmap

    def draw_plane(self, suffix='', epoch=-1, to_unscaled_coord=True, plane_id=None, save_mesh=True):
        plane_normal, _, plane_center, plane_radii, plane_rot_q, _, _ = self.get_plane_geometry()
        mesh_n = plot_rectangle_planes(
            plane_center, plane_normal, plane_radii, plane_rot_q, 
            epoch=epoch, 
            suffix='%s'%(suffix), 
            to_unscaled_coord=to_unscaled_coord, 
            pose_cfg=self.pose_cfg, 
            out_path=self.plot_dir if save_mesh else None,
            plane_id=plane_id, 
            color_type='normal')
        mesh_p = plot_rectangle_planes(
            plane_center, plane_normal, plane_radii, plane_rot_q, 
            epoch=epoch, 
            suffix='%s'%(suffix), 
            to_unscaled_coord=to_unscaled_coord, 
            pose_cfg=self.pose_cfg, 
            out_path=self.plot_dir if save_mesh else None,
            plane_id=plane_id, 
            color_type='prim')
        return mesh_n, mesh_p