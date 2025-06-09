<div align="center">
<h1>PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes</h1>

<a href="https://arxiv.org/abs/2412.03451"><img src="https://img.shields.io/badge/arXiv-2412.03451-b31b1b" alt="arXiv"></a> <a href="https://icetttb.github.io/PlanarSplatting/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a> <a href=""><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue"></a>

[Bin Tan<sup>1</sup>](https://icetttb.github.io/), [Rui Yu<sup>2</sup>](https://ruiyu0.github.io/), [Yujun Shen<sup>1</sup>](https://shenyujun.github.io/), [Nan Xue<sup>1</sup>](https://xuenan.net/)

<sup>1</sup>Ant Group  <sup>2</sup>University of Louisville

</div>

Official implementation of **PlanarSplatting** (CVPR 2025 Hightlight) 
--
An **ultrafast** per-scene optimization method built upon planar primitives for indoor planar surface reconstruction.

<div align="center">
  <img src="./assets/pipeline.png" alt="teaser" width="800" />
</div>

## 📖 Overview
We present PlanarSplatting, an ultra-fast and accurate surface reconstruction approach for multi-view indoor images. We take the 3D planes as the main objective due to their compactness and structural expressiveness in indoor scenes, and develop an explicit optimization framework that learns to fit the expected surface of indoor scenes by splatting the 3D planes into 2.5D depth and normal maps.


## ⚙️ Installation
### 1. Clone PlanarSplatting
```
git clone url/to/planarspaltting --recursive 
```
### 2. Create the enviroment
```
conda create -n planarSplatting python=3.10
conda activate planarSplatting

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install submodules/diff-rect-rasterization
pip install submodules/quaternion-utils

# for running on self-captured images
pip install -e submodules/vggt/
```

## 🎯 Quick Start
You can run PlanarSpaltting with an interactive demo for your own data as follows:
```
python run_gradio_demo.py
```
Here, we use state-of-the-art [VGGT](https://github.com/facebookresearch/vggt) for relative pose and depth estimation and [Metric3D v2](http://github.com/YvanYin/Metric3D) for normal estimation.
<div align="center">
  <img src="./assets/demo.png" alt="demo" width="800" />
</div>

Alternatively, you can also run PlanarSpaltting without a GUI as follows:
```shell
python run_demo.py --data_path path/to/images
```

## 🧪 Run on the ScanNetV2 scenes
### 1. Download ScanNetv2
Please follow instructions reported in [SimpleRecon](https://github.com/nianticlabs/airplanes/blob/main/README.md) to download and extract ScanNetV2 scenes. The extacted data format should be like:

    data/ScanNetV2
        scans/
            scene0000_00/
                scene0000_00_vh_clean_2.ply (gt mesh)
                sensor_data/
                    frame-000000.pose.txt
                    frame-000000.color.jpg 
                scene0000.txt (scan metadata and image sizes)
                intrinsic/
                    intrinsic_depth.txt
                    intrinsic_color.txt
            scene0000_01/
            ....

### 2. Process scenes for PlanarSplatting
``` shell
cd planarsplat/data_process/scannetv2
python scannetv2_sample_rgb_intrinsic_pose.py \
    --data_path ../../data/ScanNetV2/ \
    --out_path ../../data/ScanNetV2/processed_scans/ \
    --scene_id scene0488_00
```

### 3. Run one one scene
We follow the evaluation methodology from [AirPlanes](https://github.com/nianticlabs/airplanes) on the ScanNetV2 dataset. For more details, please refer to the same [AirPlanes](https://github.com/nianticlabs/airplanes) repository.
```shell
cd planarsplat
python run/runner.py  \
    --base_conf confs/base_conf_planarSplatCuda.conf \
    --conf confs/scannetv2_train.conf \
    --gpu 0 \
    --scan_id scene0488_00;
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for runner.py</span></summary>

  #### --base_conf
  Path to the base config file.
  #### --conf
  Path to the scene specific config file.
  #### --gpu
  Index of used GPU.
  #### --scan_id
  Name of the scan.
  #### --is_continue
  Flag to resume from the latest optimization.

</details>
<br>

## 📜 Citation
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{PlanarSplatting2024,
    title   = {PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes},
    author  = {Tan, Bin and Yu, Rui and Shen, Yujun and Xue, Nan},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year    = {2025}
}
```

## 🙏 Acknowledgement
PlanarSplatting is constructed on the outstanding open-source projects. We are extremely grateful for the contributions of these projects and their communities, whose hard work has greatly propelled the development of the field and enabled our work to be realized.

- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
- [AirPlanes](https://github.com/nianticlabs/airplanes)
- [VGGT](https://github.com/facebookresearch/vggt)
- [Metric3D v2](https://github.com/YvanYin/Metric3D/tree/main)
- [Omnidata](https://github.com/EPFL-VILAB/omnidata)
