import sys
sys.path.append('../planarsplat')
import argparse
import os
from utils.misc_util import fix_seeds, get_class
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/example.conf', help='path to config file')
    parser.add_argument('--base_conf', type=str, default='confs/base_conf_planarSplatCuda.conf', help='path to base config file')
    parser.add_argument('--run_task', type=str, default='train', help='run task: train, eval')
    parser.add_argument('--gpu', type=str, default='0,', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='if set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--ckpt', default='latest', type=str, help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--cancel_vis', default=False, action="store_true", help='cancel visualization durning training')
    parser.add_argument('--scan_id', type=str, default='-1', help='If set, taken to be the scan id.')
    args = parser.parse_args()

    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # load base conf at first
    base_conf = ConfigFactory.parse_file(args.base_conf)
    scene_conf = ConfigFactory.parse_file(args.conf)
    cfg = ConfigTree.merge_configs(base_conf, scene_conf)

    # process scan_id if needed
    scan_id = cfg.get_string('dataset.scan_id', default='-1')
    if scan_id == '-1':
        assert args.scan_id != '-1', "scan_id should be given!"
        cfg.put('dataset.scan_id', args.scan_id)
    
    # get name of experiment folder
    exps_folder_name = cfg.get_string('train.exps_folder_name', default='exps_result')

    if args.run_task == 'train':
        fix_seeds()
        runner = get_class(cfg.get_string('train.train_runner_class'))(
                                    conf=cfg,
                                    batch_size=1,
                                    exps_folder_name=exps_folder_name,
                                    is_continue=args.is_continue,
                                    timestamp=args.timestamp,
                                    checkpoint=args.ckpt,
                                    do_vis=not args.cancel_vis,
                                    scan_id=args.scan_id,
                                    )
    else:
        raise ValueError('Undefined run task!')
    
    runner.run()
