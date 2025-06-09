import os
import torch
import random
import numpy as np
from loguru import logger
from datetime import datetime
import shutil
from glob import glob
from PIL import Image
from pyhocon import HOCONConverter
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')


def fix_seeds(random_seed=42, use_deterministic_algorithms=True):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(use_deterministic_algorithms)

def setup_logging(log_file='my_log.log'):
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} {file}:{line} {level} {message}", level="DEBUG")


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_train_param(opt, conf):
    expname = conf.get_string('train.expname')    
    scan_id = opt['scan_id'] if opt['scan_id'] != '-1' else conf.get_string('dataset.scan_id', default='-1')
    if scan_id != '-1':
        expname = expname + '_{0}'.format(scan_id)
    if opt['is_continue']:
        if opt['timestamp'] == 'latest':
            # if os.path.exists(os.path.join('../', opt['exps_folder_name'], expname)):
            if os.path.exists(os.path.join(root, opt['exps_folder_name'], expname)):
                # timestamps = os.listdir(os.path.join('../', opt['exps_folder_name'], expname))
                timestamps = os.listdir(os.path.join(root, opt['exps_folder_name'], expname))
                if (len(timestamps)) == 0:
                    raise ValueError('There are no experiments in the target folder!')
                else:
                    timestamp = sorted(timestamps)[-1]  # use latest timestamp
                    is_continue = True
            else:
                raise ValueError('Target folder does not exist!')
        else:
            # if os.path.exists(os.path.join('../', opt['exps_folder_name'], expname, opt['timestamp'])):
            if os.path.exists(os.path.join(root, opt['exps_folder_name'], expname, opt['timestamp'])):
                is_continue = True
                timestamp = opt['timestamp']
            else:
                raise ValueError('Target folder does not exist!')
    else:
        is_continue = False
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    
    return expname, scan_id, timestamp, is_continue

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths


def load_rgb(path, normalize_rgb = False):
    img = Image.open(path)
    img = np.array(img)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.
    if normalize_rgb: #  [0,1] ---> [-1,1]
        img -= 0.5
        img *= 2.
    return img

def prepare_folders(kwargs, expname, timestamp, suffix=''):
    # =======================================  create experiment folder
    exps_folder_name = kwargs['exps_folder_name']
    expdir = os.path.join(root, exps_folder_name, expname, timestamp)
    mkdir_ifnotexists(expdir)
    # =======================================  create plot folder
    if len(suffix) > 0:
        plane_plots_dir = os.path.join(expdir, f'plane_plots_{suffix}')
    else:
        plane_plots_dir = os.path.join(expdir, 'plane_plots')
    mkdir_ifnotexists(plane_plots_dir)
    # =======================================  create checkpoint folder
    checkpoints_path = os.path.join(expdir, 'checkpoints')
    model_subdir = "Parameters"
    mkdir_ifnotexists(os.path.join(checkpoints_path, model_subdir))

    return expdir, plane_plots_dir, checkpoints_path, model_subdir

def prepare_rec_folders(rec_folder_name="rec_result", dataset="scannetv2", scan_id='scenexxxx_xx'):
    # =======================================  create experiment folder
    exps_folder_name = "rec_result"
    # recdir = os.path.join('../', rec_folder_name, dataset, scan_id)
    recdir = os.path.join(root, rec_folder_name, dataset, scan_id)
    mkdir_ifnotexists(recdir)
    return recdir

def save_config_files(expdir, conf):
    timestamp_ = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    cfg_path = os.path.join(expdir, f'run_conf_train_{timestamp_}.conf')
    with open(cfg_path, "w") as fd:
        fd.write(HOCONConverter.to_json(conf))
        
    source_tain_file = os.path.join(*(conf.get_string('train.train_runner_class').split('.')[:-1])) + '.py'
    if os.path.exists(source_tain_file):
        shutil.copy(source_tain_file, os.path.join(expdir, f'trainer_{timestamp_}.py'))