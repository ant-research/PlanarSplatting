import gradio as gr
import numpy as np
import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_vggt import run_vggt
from utils_demo.run_metric3d import extract_mono_geo_demo
from utils.misc_util import get_class
import shutil
from gradio_rerun import Rerun
import rerun as rr
import rerun.blueprint as rrb
import math
from tqdm import tqdm
from utils.loss_util import normal_loss, metric_depth_loss
from utils.trainer_util import plot_plane_img
import random

repo_path = os.path.dirname(__file__)
gradio_tmp_path = os.path.join(repo_path, 'tmp_gradio')
os.environ["GRADIO_TEMP_DIR"] = gradio_tmp_path

if os.path.exists(gradio_tmp_path):
    shutil.rmtree(gradio_tmp_path)
os.makedirs(gradio_tmp_path, exist_ok=False)  # Create the directory if it doesn't exist

def get_recording(recording_id: str) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)

def generate_random_string(length=3, characters='abcdef123'):
    return ''.join(random.choice(characters) for _ in range(length))

def opt_one_iter(runner, iter, weight_decay_list, view_info_list, log_freq=10):
    # ======================================= process planes
    if iter > runner.coarse_stage_ite and iter % runner.process_plane_freq_ite==0:  
        runner.net.regularize_plane_shape()
        runner.net.prune_small_plane()
        if iter > runner.split_start_ite and iter <= runner.max_total_iters - 1000:
            ori_num = runner.net.planarSplat.get_plane_num()
            runner.net.split_plane()
            new_num = runner.net.planarSplat.get_plane_num()
            print(f'plane num: {ori_num} ---> {new_num}')
    # ======================================= get view info
    if not view_info_list:
        view_info_list = runner.dataset.view_info_list.copy()
    view_info = view_info_list.pop(0)
    raster_cam_w2c = view_info.raster_cam_w2c
    # ======================================= zero grad
    runner.net.optimizer.zero_grad()
    #  ======================================= plane forward
    allmap = runner.net.planarSplat(view_info,iter)
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
    loss_plane = (loss_plane_depth * 1.0) * runner.weight_plane_depth \
                + (loss_plane_normal_l1 + loss_plane_normal_cos) * runner.weight_plane_normal
    loss_final += loss_plane * decay * 2.0

    # ======================================= backward & update plane denom & update learning rate
    loss_final.backward()
    runner.net.optimizer.step()
    runner.net.update_grad_stats()
    runner.net.regularize_plane_shape(False)
    image_index = view_info.index
    runner.dataset.view_info_list[image_index].plane_depth = depth.detach().clone()
    
    if iter > 0 and iter % runner.check_vis_freq_ite == 0:
        runner.check_plane_visibility_cuda()
    
    if iter % runner.plot_freq == 0:
        runner.net.regularize_plane_shape()
        runner.net.eval()
        runner.net.planarSplat.draw_plane(epoch=iter)
        plot_plane_img(runner)
        runner.net.train()

    if iter % log_freq == 0:
        mesh_n, mesh_p = runner.net.planarSplat.draw_plane(epoch=iter, save_mesh=False)
    else:
        mesh_n, mesh_p = None, None
    
    return view_info_list, mesh_n, mesh_p

def run_model(image_paths, depth_conf_thresh, iteration_num, init_prim_num, prim_split_thresh, plot_freq):
    exp_name = generate_random_string()
    # rec = get_recording(recording_id='ex1')
    rec = get_recording(recording_id=exp_name)
    stream = rec.binary_stream()
    blueprint = rrb.Blueprint(
        # rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="mesh/normal"),
                rrb.Spatial3DView(origin="mesh/prim"),
                rrb.Spatial3DView(origin="mesh/merged"),
            ),
        # ),
        collapse_panels=True,
    )
    rec.send_blueprint(blueprint)
    rec.log("mesh/normal", rr.Clear(recursive=True))
    rec.log("mesh/prim", rr.Clear(recursive=True))
    rec.log("mesh/merged", rr.Clear(recursive=True))
    rec.reset_time()
    yield stream.read(), ''
    rec.log("mesh/normal", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rec.log("mesh/prim", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rec.log("mesh/merged", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rec.set_time("iteration", sequence=0)
    yield stream.read(), ''

    status = "<div style='text-align: center; font-size: 24px;'>Processing Input Images...</div>"
    yield None, status
    out_dir = gradio_tmp_path
    # run vggt
    data = run_vggt(image_paths, out_dir, depth_conf_thresh=depth_conf_thresh)
    # run metric3dv2
    _, normal_maps_list = extract_mono_geo_demo(data['color'], data['intrinsics'])
    data['normal'] = normal_maps_list
    # run planarSplatting
    ## load conf
    base_conf = ConfigFactory.parse_file('planarsplat/confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file('utils_demo/demo.conf')
    conf = ConfigTree.merge_configs(base_conf, demo_conf)
    conf.put('train.exps_folder_name', out_dir)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.img_res', img_res)
    conf.put('train.max_total_iters', iteration_num)
    conf.put('plane_model.init_plane_num', init_prim_num)
    conf.put('plane_model.split_thres', prim_split_thresh)
    ## run optimization
    exps_folder_name = conf.get_string('train.exps_folder_name')
    planarSplatting_runner = get_class(conf.get_string('train.train_runner_class'))(
                                    conf=conf,
                                    batch_size=1,
                                    exps_folder_name=exps_folder_name,
                                    is_continue=False,
                                    timestamp='latest',
                                    checkpoint='latest',
                                    do_vis=True,
                                    scan_id='-1',
                                    data=data,
                                    )
    if planarSplatting_runner.start_iter >= planarSplatting_runner.max_total_iters:
        return
    weight_decay_list = []
    for i in tqdm(range(planarSplatting_runner.max_total_iters+1), desc="generating sampling idx list..."):
        weight_decay_list.append(max(math.exp(-i / planarSplatting_runner.max_total_iters), 0.1))
    planarSplatting_runner.net.train()
    if planarSplatting_runner.iter_step == 0:
        planarSplatting_runner.check_plane_visibility_cuda()  
    view_info_list = None
    progress_bar = tqdm(range(planarSplatting_runner.start_iter, planarSplatting_runner.max_total_iters+1), desc="Training progress")
    max_iter = planarSplatting_runner.max_total_iters
    for iter in range(planarSplatting_runner.start_iter, max_iter + 1):
        planarSplatting_runner.iter_step = iter

        view_info_list, mesh_n, mesh_p = opt_one_iter(planarSplatting_runner, iter, weight_decay_list, view_info_list, log_freq=plot_freq)

        with torch.no_grad():
            # Progress bar
            plane_num = planarSplatting_runner.net.planarSplat.get_plane_num()
            if iter % 10 == 0:
                loss_dict = {
                    "Planes": f"{plane_num}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iter == planarSplatting_runner.max_total_iters:
                progress_bar.close()
        opt_status = f"<div style='text-align: center; font-size: 24px;'>Optimizing ({iter}/{max_iter})...</div>"
        if mesh_p is not None and mesh_n is not None:
            vertex_positions = np.asarray(mesh_p.vertices)
            vertex_colors = np.clip(np.asarray(mesh_p.vertex_colors) * 255, a_min=0, a_max=255).astype(np.uint8)
            triangle_indices = np.asarray(mesh_p.triangles)
            rec.set_time("iteration", sequence=iter)
            rec.log(
                "mesh/prim",
                rr.Mesh3D(
                    vertex_positions=vertex_positions.tolist(),
                    vertex_colors=vertex_colors.tolist(),
                    triangle_indices=triangle_indices.tolist(),
                ),
            )

            vertex_positions = np.asarray(mesh_n.vertices)
            vertex_colors = np.clip(np.asarray(mesh_n.vertex_colors) * 255, a_min=0, a_max=255).astype(np.uint8)
            triangle_indices = np.asarray(mesh_n.triangles)
            rec.log(
                "mesh/normal",
                rr.Mesh3D(
                    vertex_positions=vertex_positions.tolist(),
                    vertex_colors=vertex_colors.tolist(),
                    triangle_indices=triangle_indices.tolist(),
                ),
            )
            yield stream.read(), opt_status
        else:
            yield None, opt_status
    ## merge prims
    status = "<div style='text-align: center; font-size: 24px;'>Merging...</div>"
    yield None, status
    rec.set_time("iteration", sequence=planarSplatting_runner.max_total_iters)
    planarSplatting_runner.check_plane_visibility_cuda()
    planar_mesh = planarSplatting_runner.merger(save_mesh=False)
    simplified_mesh = planar_mesh.simplify_quadric_decimation(15000)
    vertex_positions = np.asarray(simplified_mesh.vertices)
    vertex_colors = np.clip(np.asarray(simplified_mesh.vertex_colors) * 255, a_min=0, a_max=255).astype(np.uint8)
    triangle_indices = np.asarray(simplified_mesh.triangles)
    rec.log(
        "mesh/merged",
        rr.Mesh3D(
            vertex_positions=vertex_positions.tolist(),
            vertex_colors=vertex_colors.tolist(),
            triangle_indices=triangle_indices.tolist(),
        ),
    )
    status = "<div style='text-align: center; font-size: 24px;'>Finished</div>"
    yield stream.read(), status

def show_image(image_paths, idx=0):
    image_path = image_paths[idx]
    return image_path

def load_test_images(test_images_dir):
    return [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]


TEST_IMAGES_DIR_1 = 'examples_gradio'
img_paths_case1 = load_test_images(TEST_IMAGES_DIR_1)
test_cases = {
    "Test Case 1": {
        "files": img_paths_case1,
    },
}

def load_test_case(test_case_name):
    image_files = test_cases[test_case_name]['files']
    return image_files

with gr.Blocks() as demo:
    gr.Markdown("### Upload Images")

    with gr.Tab(label="Upload Images"):
        test_case_dropdown = gr.Dropdown(
            label="Select Test Case",
            choices=list(test_cases.keys()),
            value='',
            interactive=True
        )
        with gr.Column():
            multi_files = gr.File(file_count="multiple")
    with gr.Row():
        depth_conf_thresh = gr.Slider(label="depth_conf_thresh", value=2.0, minimum=0.1, maximum=20, step=0.1)
        iteration_num = gr.Slider(label="iteration_num", value=2500, minimum=1000, maximum=10000, step=500)
    with gr.Row():
        init_prim_num = gr.Slider(label="init_prim_num", value=1500, minimum=500, maximum=3000, step=100)
        prim_split_thresh = gr.Slider(label="prim_split_thresh", value=0.2, minimum=0.05, maximum=10, step=0.05)
    with gr.Row():
        plot_freq = gr.Slider(label="plot_freq", value=200, minimum=1, maximum=200, step=1)
    run_button = gr.Button("Run")
    status_output = gr.Markdown("", label="Running Status")
    with gr.Row():
        viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
        )
    test_case_dropdown.change(
        fn=load_test_case,
        inputs=test_case_dropdown,
        outputs=multi_files,
    )
    run_button.click(fn=run_model, 
                     inputs=[multi_files, depth_conf_thresh, iteration_num, init_prim_num, prim_split_thresh, plot_freq],
                     outputs=[viewer, status_output])

demo.launch()
