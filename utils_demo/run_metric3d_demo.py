# This code is modified from
# https://github.com/YvanYin/Metric3D/blob/main/hubconf.py


dependencies = ['torch', 'torchvision']

import os
import torch
try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction

import sys
sys.path.append('submodules/Metric3D')
print(sys.path)

from mono.model.monodepth_model import get_configured_monodepth_model
metric3d_dir = os.path.dirname(__file__)

from torchvision import transforms
trans_topil = transforms.ToPILImage()


MODEL_TYPE = {
  'ConvNeXt-Tiny': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
  },
  'ConvNeXt-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
  },
  'ViT-Small': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
  },
  'ViT-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
  },
  'ViT-giant2': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
  },
}


def metric3d_convnext_tiny(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_convnext_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_small(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_giant2(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model


if __name__ == '__main__':
    import cv2
    import numpy as np
    import argparse
    import glob
    import matplotlib.pyplot as plt
    from torchvision import transforms

    trans_topil = transforms.ToPILImage()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--data_path", type=str, default='../data', help='root path to data')
    parser.add_argument("--scene_id", type=str, default='', help='name of one scene')
    parser.add_argument("--scene_list", type=str, default='', help='path to the scene list file')
    parser.add_argument("--model_cache_path", type=str, default='', help='')
    parser.add_argument("--model_type", type=str, default='metric3d_vit_large', help='model used for inference')

    args = parser.parse_args()

    # build model
    if len(args.model_cache_path) == 0:
      model = torch.hub.load('yvanyin/metric3d', args.model_type, pretrain=True)
    else:
      model = torch.hub.load(args.model_cache_path, args.model_type, trust_repo=True, source='local', pretrain=True)
    model.cuda().eval()

    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f'The path {data_path} does not exist')
    scans_path = data_path

    # get scene ids
    if args.scene_id:
        scenes_list = [args.scene_id]
    elif args.scene_list:
        scenes_list = []
        with open(args.scene_list, 'r') as file:
            lines = file.readlines()
            for line in lines:
                cleaned_line = line.strip()
                scenes_list.append(cleaned_line)
    else:
        raise ValueError("One of the [scene_id, scene_list] should be given!")

    for scene_id in scenes_list:
        scene_path = os.path.join(scans_path, scene_id)
        img_path = os.path.join(scene_path, f'images')
        if not os.path.exists(img_path):
            raise ValueError(f'The path {img_path} does not exist')

        out_path = os.path.join(scene_path, f'metric3dv2')
        os.makedirs(out_path, exist_ok=True)
        intrinsic_matrix = np.loadtxt(os.path.join(scene_path, f'intrinsic/intrinsic.txt'))

        img_list = glob.glob(img_path+'/*')
        img_list = sorted(img_list)
        for cur_img_path in img_list:
          intrinsic = [intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]]
          #### prepare data
          rgb_file = cur_img_path
          rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

          #### ajust input size to fit pretrained model
          # keep ratio resize
          input_size = (616, 1064) # for vit model
          # input_size = (544, 1216) # for convnext model
          h, w = rgb_origin.shape[:2]
          scale = min(input_size[0] / h, input_size[1] / w)
          rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
          # remember to scale intrinsic, hold depth
          intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
          # padding to input_size
          padding = [123.675, 116.28, 103.53]
          h, w = rgb.shape[:2]
          pad_h = input_size[0] - h
          pad_w = input_size[1] - w
          pad_h_half = pad_h // 2
          pad_w_half = pad_w // 2
          rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
          pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

          #### normalize
          mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
          std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
          rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
          rgb = torch.div((rgb - mean), std)
          rgb = rgb[None, :, :, :].cuda()

          ###################### canonical camera space ######################
          # inference
          with torch.no_grad():
              pred_depth, confidence, output_dict = model.inference({'input': rgb})

          # un pad
          pred_depth = pred_depth.squeeze()
          pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
          
          # upsample to original size
          pred_depth_resized = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
          ###################### canonical camera space ######################

          #### de-canonical transform
          canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
          pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
          pred_depth = torch.clamp(pred_depth, 0, 300)
          
          output_file_name = os.path.splitext(os.path.basename(cur_img_path))[0]
          save_path = os.path.join(out_path, f'{output_file_name}_depth.png')

          pred_depth[pred_depth > 10] = 0.
          np.save(save_path.replace('.png', '.npy'), pred_depth.detach().cpu().numpy())
          plt.imsave(save_path, pred_depth.detach().cpu().squeeze(), cmap='viridis')
          print(f'save to {save_path}')

          #### normal are also available
          if 'prediction_normal' in output_dict: # only available for Metric3Dv2, i.e. vit model
              pred_normal = output_dict['prediction_normal'][:, :3, :, :]
              normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
              # un pad and resize to some size if needed
              pred_normal = pred_normal.squeeze()
              pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
              
              pred_normal_resized = torch.nn.functional.interpolate(pred_normal[None, :, :, :], rgb_origin.shape[:2], mode='nearest').squeeze()
              pred_normal_resized_normalized = (pred_normal_resized + 1) / 2.0

              save_path = os.path.join(out_path, f'{output_file_name}_normal.png')
              np.save(save_path.replace('.png', '.npy'), pred_normal_resized_normalized.detach().cpu().numpy())
              trans_topil(pred_normal_resized_normalized).save(save_path)
              print(f'save to {save_path}')
              # you can now do anything with the normal
              # such as visualize pred_normal
              # import pdb; pdb.set_trace()
              # pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
              # pred_normal_vis = (pred_normal_vis + 1) / 2
              # cv2.imwrite('normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))
          else:
              raise