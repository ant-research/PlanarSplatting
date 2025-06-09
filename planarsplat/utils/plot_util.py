import os
import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from utils.model_util import quat_to_rot

class random_color(object):
    def __init__(self, color_num=20000):
        num_of_colors=color_num
        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
             for j in range(num_of_colors)]

    def __call__(self, ret_n = 10):
        assert len(self.colors) > ret_n
        ret_color = np.zeros([ret_n, 3])
        for i in range(ret_n):
            hex_color = self.colors[i][1:]
            ret_color[i] = np.array([int(hex_color[j:j + 2], 16) for j in (0, 2, 4)])
        ret_color[0] *= 0
        return ret_color

def plot_rectangle_planes(plane_centers, plane_normals, plane_radii, rot_q, epoch=-1, suffix='', to_unscaled_coord=True, pose_cfg=None, out_path=None, plane_id=None, color_type=''):
    plane_normals_standard = torch.zeros_like(plane_normals)
    plane_normals_standard[..., -1] = 1
    if plane_radii.shape[-1] == 2:
        radii_x_p = plane_radii[..., 0]  # n
        radii_y_p = plane_radii[..., 1]  # n
        radii_x_n = plane_radii[..., 0]  # n
        radii_y_n = plane_radii[..., 1]  # n
    elif plane_radii.shape[-1] == 4:
        radii_x_p = plane_radii[..., 0]  # n
        radii_y_p = plane_radii[..., 1]  # n
        radii_x_n = plane_radii[..., 2]  # n
        radii_y_n = plane_radii[..., 3]  # n
    else:
        raise NotImplementedError
    
    zero_tmp = torch.zeros_like(radii_x_p)  # n
    v1 = torch.stack([radii_x_p, radii_y_p, zero_tmp], dim=-1)  # n, 3
    v2 = torch.stack([-radii_x_n, radii_y_p, zero_tmp], dim=-1)  # n, 3
    v3 = torch.stack([-radii_x_n, -radii_y_n, zero_tmp], dim=-1)  # n, 3
    v4 = torch.stack([radii_x_p, -radii_y_n, zero_tmp], dim=-1)  # n, 3

    vertices_standard = torch.stack([v1, v2, v3, v4], dim=1)  # n, 4, 3
    rot_q = F.normalize(rot_q, dim=-1)  # n, 4
    rot_matrix = quat_to_rot(rot_q)  # n, 3, 3
    vertices_transformed = torch.bmm(rot_matrix, vertices_standard.permute(0, 2, 1)).permute(0, 2, 1) + plane_centers[:, None]  # n, 4, 3
    vertices_all = vertices_transformed.reshape(-1, 3).detach().cpu().numpy()  # 4n, 3
    
    if to_unscaled_coord:
        assert pose_cfg is not None
        vertices_all /= pose_cfg.scale
        vertices_all += pose_cfg.offset
    else:
        suffix = suffix + '_normalized'

    N = vertices_all.shape[0] // 4
    triangle_mesh = o3d.geometry.TriangleMesh()
    triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices_all)

    if color_type == 'normal':
        normal_color = (plane_normals + 1.) / 2.
        normal_color = normal_color.reshape(-1, 1, 3).repeat(1, 4, 1)
        colors = normal_color.detach().cpu().numpy().reshape(-1, 3)
        triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        suffix = suffix + '_colorNormal'
    elif color_type == 'prim':
        if plane_id is None:
            plane_id = torch.arange(plane_centers.shape[0])
        plane_id = plane_id.reshape(-1, 1).repeat(1, 4).int().cuda()
        color_vis = random_color()
        colorMap_vis = color_vis(plane_id.unique().max().item() + 1)
        colors = colorMap_vis[plane_id.detach().cpu().numpy().reshape(-1)]
        colors = colors.astype(np.float64) / 255.
        triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        suffix = suffix + '_colorPrim'
    else:
        pass

    idx_v1 = np.arange(N).reshape(-1, 1) * 4
    idx_v2 = idx_v1 + 1
    idx_v3 = idx_v1 + 2
    idx_v4 = idx_v1 + 3

    triangle_vIdx1 = np.concatenate([idx_v1, idx_v2, idx_v3], axis=-1)  # 2n, 3
    triangle_vIdx2 = np.concatenate([idx_v3, idx_v4, idx_v1], axis=-1)  # 2n, 3
    triangle_vIdx = np.concatenate([triangle_vIdx1, triangle_vIdx2], axis=0)  # 4n, 3
    triangle_mesh.triangles = o3d.utility.Vector3iVector(triangle_vIdx)

    if out_path is not None and os.path.exists(out_path):
        if epoch == -1:
            os.makedirs(out_path, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(out_path, "planarSplat_%s.ply" % (suffix)), triangle_mesh)
        else:
            str_epoch = ('%d' % (epoch)).zfill(3)
            os.makedirs(out_path, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(out_path, "planarSplat_%s_%s.ply" % (str_epoch, suffix)), triangle_mesh)
    return triangle_mesh

def plot_depth(depth, out_path):
    if not isinstance(depth, np.ndarray):
        depth_array = depth.cpu().numpy()
    else:
        depth_array = depth
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
    plt.axis('off')
    # convert to 8-bit image
    depth_image = (depth_normalized * 255).astype(np.uint8)
    plt.imsave(out_path, depth_image, cmap='plasma_r')

def plot_rgb(rgb, out_path):
    if not isinstance(rgb, np.ndarray):
        rgb_array = rgb.cpu().numpy()
    else:
        rgb_array = rgb
    plt.axis('off')
    # convert to 8-bit image
    rgb_image = (rgb_array * 255).astype(np.uint8)
    plt.imsave(out_path, rgb_image)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)