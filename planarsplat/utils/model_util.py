import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import quaternion
import math
import cv2

def get_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def quat_to_rot(q):
    assert isinstance(q, torch.Tensor)
    assert q.shape[-1] == 4
    if q.dim() == 1:
        q = q.unsqueeze(0)  # 1, 4
    elif q.dim() == 2:
        pass  # bs, 4
    else:
        raise NotImplementedError

    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    # R = torch.zeros((batch_size, 3, 3), device='cuda')
    qr = q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q

def quaternion_mult(q1, q2):
    '''
    q1 x q2

    q1 = w1+i*x1+j*y1+k*z1
    q2 = w2+i*x2+j*y2+k*z2
    q1*q2 =
     (w1w2 - x1x2 - y1y2 - z1z2)
    +(w1x2 + x1w2 + y1z2 - z1y2)i
    +(w1y2 - x1z2 + y1w2 + z1x2)j
    +(w1z2 + x1y2 - y1x2 + z1w2)k

    :param q1:
    :param q2:
    :return:
    '''

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    w1 = q1[:, 0]  # bs
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]

    w2 = q2[:, 0]  # bs
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q = torch.stack([w, x, y, z], dim=-1)

    return q

def get_rotation_quaternion_of_normal(plane_normal_init, standard_normal=None):
    if standard_normal is None:
        standard_normal = torch.tensor([0., 0., 1.]).reshape(1, 3).expand(plane_normal_init.shape[0], 3).to(plane_normal_init.device)
    angle_diff = torch.acos((standard_normal * plane_normal_init).sum(dim=-1).clamp(-1, 1)).reshape(-1, 1)
    rot_axis = torch.cross(standard_normal, plane_normal_init, dim=-1)  # n_plane, 3
    rot_axis = F.normalize(rot_axis, dim=-1)
    rot_vec = (rot_axis * angle_diff).cpu().numpy()
    rot_q = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vec))
    rot_q = torch.from_numpy(rot_q).float()
    return rot_q

def get_rotation_quaternion_of_xyAxis(plane_num, angle=None):
    rot_axis = torch.tensor([0., 0., 1.]).reshape(1, 3)
    if angle is None:
        rand_angle = (torch.rand(plane_num, 1) - 0.5) * 5. / 180. * np.pi
    else:
        rand_angle = angle / 180. * np.pi
    rot_vec = (rot_axis * rand_angle.cpu()).numpy()
    rot_q = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vec))
    rot_q = torch.from_numpy(rot_q).float()
    return rot_q

def get_overlapped_mask(plane_center, plane_normal, plane_offset, plane_radii, plane_xAxis, plane_yAxis, normal_thres=15, dist_thres=0.02):
    plane_num = plane_center.shape[0]
    corner1 = plane_center + plane_radii[:, 0:1] * plane_xAxis + plane_radii[:, 1:2] * plane_yAxis  # n, 3
    corner2 = plane_center + plane_radii[:, 0:1] * plane_xAxis - plane_radii[:, 1:2] * plane_yAxis
    corner3 = plane_center - plane_radii[:, 0:1] * plane_xAxis + plane_radii[:, 1:2] * plane_yAxis
    corner4 = plane_center - plane_radii[:, 0:1] * plane_xAxis - plane_radii[:, 1:2] * plane_yAxis

    pts = torch.stack([corner1, corner2, corner3, corner4], dim=1)  # n, 4, 3
    nc = pts.shape[1]
    pts = pts.reshape(-1, 3)  # n * 4, 3

    dists = compute_pts2planes_dist(pts, plane_normal, plane_offset)  # n * 4, n, 1
    pts_proj = pts.reshape(-1, 1, 3) - dists * plane_normal.reshape(1, -1, 3)  # n * 4, n, 3 

    vec = pts_proj - plane_center.reshape(1, -1, 3)  # n * 4, n, 3
    if_in_x = ((vec * plane_xAxis.reshape(1, -1, 3)).sum(dim=-1).abs() -  plane_radii[:, 0:1].reshape(1, -1)) <= 0.0005   # n * 4, n
    if_in_y = ((vec * plane_yAxis.reshape(1, -1, 3)).sum(dim=-1).abs() -  plane_radii[:, 1:2].reshape(1, -1)) <= 0.0005   # n * 4, n

    if_in = if_in_x & if_in_y   # n * 4, n
    if_in = if_in.reshape(-1, nc, plane_num)   # n, 4, n
    if_in = if_in.sum(dim=1) == nc  # n~, n

    normal_diff = torch.acos((plane_normal.reshape(-1, 1, 3) * plane_normal.reshape(1, -1, 3)).sum(dim=-1).clamp(-1., 1.)) * 180. / np.pi
    normal_mask = normal_diff < normal_thres

    dist_mask = (dists.reshape(-1, nc, plane_num).abs() < dist_thres).sum(dim=1) == nc

    if_in_final = if_in & normal_mask & dist_mask
    prune_mask = if_in_final.sum(dim=-1) > 1

    return prune_mask

def compute_offset(pts, normals):
    v1 = pts
    v2 = F.normalize(normals, dim=-1)
    offset = (v1 * v2).sum(-1)
    return offset

def compute_pts2planes_dist(pts, planes_normal, planes_offset):
    """
    pts: [..., 3]
    planes_normal: [N, 3]
    planes_offset: [N, 1]
    """
    planes_normal = planes_normal.unsqueeze(0)  # 1, N, 3
    planes_offset = planes_offset.unsqueeze(0)
    pts = pts.reshape(-1, 1, 3)
    pts_origin2planes = planes_normal * planes_offset  # 1, N, 3
    dist_field = ((pts - pts_origin2planes) * planes_normal).sum(-1, keepdim=True)  # npt, N, 1

    return dist_field

def get_max_and_min_radii(ite, radii_max_list, radii_min_list, radii_milestone_list):
    if ite == -1:
        max_radii = radii_max_list[-1]
        min_radii = radii_min_list[-1]
    else:
        ms_i = -1
        for ms in radii_milestone_list:
                if ite >= ms:
                    ms_i += 1
                else:
                    break
        assert ms_i >= 0
        max_radii = radii_max_list[ms_i]
        min_radii = radii_min_list[ms_i]
    
    return max_radii, min_radii

def get_plane_param_from_sphere(plane_num, radius):
    points = []
    for _ in range(plane_num):
        z = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        x *= radius
        y *= radius
        z *= radius  
        points.append([x, y, z])
    points = np.array(points)
    init_centers = torch.from_numpy(points).float()
    init_normals = F.normalize(-init_centers, dim=-1)

    init_rot_q_normal = get_rotation_quaternion_of_normal(init_normals)
    init_rot_angle_xyAxis = torch.tensor([0.]).reshape(1, 1).repeat(init_normals.shape[0], 1)
    init_rot_q_xyAxis = get_rotation_quaternion_of_xyAxis(init_normals.shape[0], angle=init_rot_angle_xyAxis)

    return init_centers, init_rot_q_normal, init_rot_q_xyAxis

def split_y_axis(selected_mask, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z):
        selected_plane_center = plane_center[selected_mask]
        selected_plane_y_axis = plane_y_axis[selected_mask]
        selected_plane_radii_x_p = plane_radii_xy_p[:, 0:1][selected_mask]
        selected_plane_radii_y_p = plane_radii_xy_p[:, 1:2][selected_mask]
        selected_plane_radii_x_n = plane_radii_xy_n[:, 0:1][selected_mask]
        selected_plane_radii_y_n = plane_radii_xy_n[:, 1:2][selected_mask]

        new1_plane_center = selected_plane_center + selected_plane_y_axis * 0.5 * selected_plane_radii_y_p
        new1_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p, 0.5 * selected_plane_radii_y_p], dim=-1)
        new1_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n, 0.5 * selected_plane_radii_y_p], dim=-1)
        
        new2_plane_center = selected_plane_center - selected_plane_y_axis * 0.5 * selected_plane_radii_y_n
        new2_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p, 0.5 * selected_plane_radii_y_n], dim=-1)
        new2_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n, 0.5 * selected_plane_radii_y_n], dim=-1)
        
        new_plane_center = torch.cat([new1_plane_center, new2_plane_center], dim=0)
        new_plane_radii_xy_p = torch.cat([new1_plane_radii_xy_p, new2_plane_radii_xy_p], dim=0)
        new_plane_radii_xy_n = torch.cat([new1_plane_radii_xy_n, new2_plane_radii_xy_n], dim=0)

        new_plane_rot_q_normal_wxy = plane_rot_q_normal_wxy[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_w = plane_rot_q_xyAxis_w[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_z = plane_rot_q_xyAxis_z[selected_mask].repeat(2, 1)

        return new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z

def split_x_axis(selected_mask, plane_x_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z):
        selected_plane_center = plane_center[selected_mask]
        selected_plane_x_axis = plane_x_axis[selected_mask]
        selected_plane_radii_x_p = plane_radii_xy_p[:, 0:1][selected_mask]
        selected_plane_radii_y_p = plane_radii_xy_p[:, 1:2][selected_mask]
        selected_plane_radii_x_n = plane_radii_xy_n[:, 0:1][selected_mask]
        selected_plane_radii_y_n = plane_radii_xy_n[:, 1:2][selected_mask]

        new1_plane_center = selected_plane_center + selected_plane_x_axis * 0.5 * selected_plane_radii_x_p
        new1_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p * 0.5, selected_plane_radii_y_p], dim=-1)
        new1_plane_radii_xy_n = torch.cat([selected_plane_radii_x_p * 0.5, selected_plane_radii_y_n], dim=-1)
        
        new2_plane_center = selected_plane_center - selected_plane_x_axis * 0.5 * selected_plane_radii_x_n
        new2_plane_radii_xy_p = torch.cat([selected_plane_radii_x_n * 0.5, selected_plane_radii_y_p], dim=-1)
        new2_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n * 0.5, selected_plane_radii_y_n], dim=-1)

        new_plane_center = torch.cat([new1_plane_center, new2_plane_center], dim=0)
        new_plane_radii_xy_p = torch.cat([new1_plane_radii_xy_p, new2_plane_radii_xy_p], dim=0)
        new_plane_radii_xy_n = torch.cat([new1_plane_radii_xy_n, new2_plane_radii_xy_n], dim=0)

        new_plane_rot_q_normal_wxy = plane_rot_q_normal_wxy[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_w = plane_rot_q_xyAxis_w[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_z = plane_rot_q_xyAxis_z[selected_mask].repeat(2, 1)

        return new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z

def split_xy_axis(selected_mask, plane_x_axis, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z):
        new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q_normal_wxy, new1_plane_rot_q_xyAxis_w, new1_plane_rot_q_xyAxis_z = split_y_axis(
            selected_mask, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)

        selected_mask_1 = torch.ones(new1_plane_center.shape[0]).cuda() > 0
        new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q_normal_wxy, new3_plane_rot_q_xyAxis_w, new3_plane_rot_q_xyAxis_z = split_x_axis(
            selected_mask_1, plane_x_axis[selected_mask].repeat(2, 1), new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q_normal_wxy, new1_plane_rot_q_xyAxis_w, new1_plane_rot_q_xyAxis_z)

        return new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q_normal_wxy, new3_plane_rot_q_xyAxis_w, new3_plane_rot_q_xyAxis_z

def get_split_mask_via_radii_grad(grads_radii, plane_radii_xy_p, plane_radii_xy_n, radii_ratio, radii_min, split_thres):
    grads_radii_max = grads_radii.max(dim=-1)[0]
    assert grads_radii_max.dim() == 1
    grads_radii_x_p = grads_radii[:, 0].contiguous()
    grads_radii_y_p = grads_radii[:, 1].contiguous()
    grads_radii_x_n = grads_radii[:, 2].contiguous()
    grads_radii_y_n = grads_radii[:, 3].contiguous()
    grad_radii_y_max = torch.max(grads_radii_y_p, grads_radii_y_n)
    grad_radii_x_max = torch.max(grads_radii_x_p, grads_radii_x_n)
    radii_x_split_mask = (plane_radii_xy_p[:, 0] + plane_radii_xy_n[:, 0]) > radii_ratio * radii_min
    radii_y_split_mask = (plane_radii_xy_p[:, 1] + plane_radii_xy_n[:, 1]) > radii_ratio * radii_min
    grad_radii_x_split_mask = grad_radii_x_max >= split_thres
    grad_radii_y_split_mask = grad_radii_y_max >= split_thres
    x_split_mask = radii_x_split_mask & grad_radii_x_split_mask
    y_split_mask = radii_y_split_mask & grad_radii_y_split_mask
    return x_split_mask, y_split_mask

def split_planes_via_mask(split_y_mask, split_x_mask, split_xy_mask, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z, rot_delta):
    new_plane_center = []
    new_plane_radii_xy_p = []
    new_plane_radii_xy_n = []
    new_plane_rot_q_normal_wxy = []
    new_plane_rot_q_xyAxis_w = []
    new_plane_rot_q_xyAxis_z = []
    new_rot_delta = []
    
    if split_y_mask.sum() > 0:
            new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q_normal_wxy, new1_plane_rot_q_xyAxis_w, new1_plane_rot_q_xyAxis_z = split_y_axis(
                split_y_mask, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)
            new_plane_center.append(new1_plane_center)
            new_plane_radii_xy_p.append(new1_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new1_plane_radii_xy_n)
            new_plane_rot_q_normal_wxy.append(new1_plane_rot_q_normal_wxy)
            new_plane_rot_q_xyAxis_w.append(new1_plane_rot_q_xyAxis_w)
            new_plane_rot_q_xyAxis_z.append(new1_plane_rot_q_xyAxis_z)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_y_mask], rot_delta[split_y_mask]], dim=0))
    if split_x_mask.sum() > 0:
            new2_plane_center, new2_plane_radii_xy_p, new2_plane_radii_xy_n, new2_plane_rot_q_normal_wxy, new2_plane_rot_q_xyAxis_w, new2_plane_rot_q_xyAxis_z = split_x_axis(
                split_x_mask, plane_xAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)
            new_plane_center.append(new2_plane_center)
            new_plane_radii_xy_p.append(new2_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new2_plane_radii_xy_n)
            new_plane_rot_q_normal_wxy.append(new2_plane_rot_q_normal_wxy)
            new_plane_rot_q_xyAxis_w.append(new2_plane_rot_q_xyAxis_w)
            new_plane_rot_q_xyAxis_z.append(new2_plane_rot_q_xyAxis_z)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_x_mask], rot_delta[split_x_mask]], dim=0))
    if split_xy_mask.sum() > 0:
            new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q_normal_wxy, new3_plane_rot_q_xyAxis_w, new3_plane_rot_q_xyAxis_z = split_xy_axis(
                split_xy_mask, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)
            new_plane_center.append(new3_plane_center)
            new_plane_radii_xy_p.append(new3_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new3_plane_radii_xy_n)
            new_plane_rot_q_normal_wxy.append(new3_plane_rot_q_normal_wxy)
            new_plane_rot_q_xyAxis_w.append(new3_plane_rot_q_xyAxis_w)
            new_plane_rot_q_xyAxis_z.append(new3_plane_rot_q_xyAxis_z)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_xy_mask], rot_delta[split_xy_mask], rot_delta[split_xy_mask], rot_delta[split_xy_mask]], dim=0))

    if len(new_plane_center) > 0:
            new_plane_center = torch.cat(new_plane_center, dim=0)
            new_plane_radii_xy_p = torch.cat(new_plane_radii_xy_p, dim=0)
            new_plane_radii_xy_n = torch.cat(new_plane_radii_xy_n, dim=0)
            new_plane_rot_q_normal_wxy = torch.cat(new_plane_rot_q_normal_wxy, dim=0)
            new_plane_rot_q_xyAxis_w = torch.cat(new_plane_rot_q_xyAxis_w, dim=0)
            new_plane_rot_q_xyAxis_z = torch.cat(new_plane_rot_q_xyAxis_z, dim=0)
            if rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)

    return new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q_normal_wxy,new_plane_rot_q_xyAxis_w,new_plane_rot_q_xyAxis_z,new_rot_delta

def get_point_cloud(depth, intrinsics, img_res, return_numpy=True):
    backproject = BackprojectDepth(1, img_res[0], img_res[1]).cuda()
    K_inv = torch.inverse(intrinsics.squeeze())[None]
    points = backproject(depth, K_inv)[0, :3, :].permute(1, 0)
    if return_numpy:
        return points.detach().cpu().numpy()
    else:
        return points