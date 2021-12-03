# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import yaml
import datetime
import platform
import numpy as np
from easydict import EasyDict as edict
from yaml import Loader, Dumper


# create dict
__C = edict()
cfg = __C

# Dataset
__C.DATASET = edict()
# Dataset Options: 'Completion3D', 'ShapeNet', 'ShapeNetCars', 'KITTI'
__C.DATASET.train_dataset = "ShapeNet"
__C.DATASET.test_dataset = "ShapeNet"
__C.DATASET.n_outpoints = 16384
__C.DATASET.num_class = 0  # overwrite by `train_MVGAN.py`

# Constants
__C.CONST = edict()
__C.CONST.device = "0"
__C.CONST.weights = None
__C.CONST.num_workers = 32
__C.CONST.n_input_points = 3000

# Directories
__C.DIR = edict()
__C.DIR.out_path = "./output"
__C.DIR.in_path = "./output/checkpoints"

# Network
__C.NETWORK = edict()
__C.NETWORK.n_sampling_points = 2048
__C.NETWORK.gridding_loss_scales = [128, 64]
__C.NETWORK.gridding_loss_alphas = [0.1, 0.01]
__C.NETWORK.n_primitives = 16
__C.NETWORK.model_type = "SpareNet"
__C.NETWORK.metric = "emd"
__C.NETWORK.encode = "Residualnet"
__C.NETWORK.use_adain = "share"
__C.NETWORK.use_selayer = False
__C.NETWORK.use_consist_loss = False

# Apex training; BUG: currently cannot support multi-gpu training
__C.APEX = edict()
__C.APEX.flag = False
__C.APEX.level = "O1"

# RENDER
__C.RENDER = edict()
__C.RENDER.img_size = 256
__C.RENDER.radius_list = [
    5.0,
    7.0,
    10.0,
]  # [5.0, 7.0, 10.0] for shapenet; [10.0, 15.0, 20.0] for completion3d
__C.RENDER.projection = "orthorgonal"  # 'orthorgonal' or 'perspective'
__C.RENDER.eyepos = 1.0  # not use
__C.RENDER.n_views = 8  # number or rendered views

# GAN training
__C.GAN = edict()
__C.GAN.use_im = True  # image-levle loss
__C.GAN.use_fm = True  # discriminator feature matching loss
__C.GAN.use_cgan = False  # projection discriminator
__C.GAN.weight_im = 1  # 1
__C.GAN.weight_fm = 1  # 1
__C.GAN.weight_l2 = 200  # 200
__C.GAN.weight_gan = 0.1  # 0.1

# Train
__C.TRAIN = edict()
__C.TRAIN.batch_size = 8
__C.TRAIN.n_epochs = 150
__C.TRAIN.save_freq = 1
__C.TRAIN.log_freq = 1
__C.TRAIN.learning_rate = 1e-4
__C.TRAIN.lr_milestones = [1000]  # don't schedule
__C.TRAIN.gamma = 0.5
__C.TRAIN.betas = (0.0, 0.9)
__C.TRAIN.weight_decay = 0

# Test config
__C.TEST = edict()
__C.TEST.mode = "default"
__C.TEST.infer_freq = 1
__C.TEST.metric_name = "EMD"  # 'EMD' or 'ChamferDistance'

# Dataset Config
__C.DATASETS = edict()
__C.DATASETS.shapenet = edict()
__C.DATASETS.shapenet.n_renderings = 8
__C.DATASETS.shapenet.n_points = 16384
# 'GRnet' or 'ShapeNet' version dataset
__C.DATASETS.shapenet.version = "GRnet"
# Training data
__C.DATASETS.shapenet.category_file_path = "./datasets/data/ShapeNet.json"
__C.DATASETS.shapenet.partial_points_path = "./datasets/data/KITTI-Cars/partial/%s/%02d.pcd"
__C.DATASETS.shapenet.complete_points_path = "./datasets/data/KITTI-Cars/complete/%s.pcd"
__C.DATASETS.shapenet.bounding_box_file_path = "./datasets/data/bboxes/bboxes/%s.txt"
__C.DATASETS.completion3d = edict()
__C.DATASETS.completion3d.category_file_path = "/path/to/datasets/data/Completion3D.json"
__C.DATASETS.completion3d.partial_points_path = "/path/to/datasets/completion3d/data/shapenet/%s/partial/%s/%s.h5"
__C.DATASETS.completion3d.complete_points_path = "/path/to/datasets/completion3d/data/shapenet/%s/gt/%s/%s.h5"
__C.DATASETS.kitti = edict()
__C.DATASETS.kitti.n_renderings = 8
__C.DATASETS.kitti.n_points = 16384
__C.DATASETS.kitti.version = "GRnet"
# Test data - Model inferencing
__C.DATASETS.kitti.category_file_path = "./datasets/data/KITTI.json"
__C.DATASETS.kitti.partial_points_path = "./datasets/data/cars/cars/%s.pcd"
# bboxes represents boundary boxes of each test image
# bboxes calculation can be seen in colab-notebooks/helper-notebooks/3D_construction_helper.ipynb
__C.DATASETS.kitti.bounding_box_file_path = "./datasets/data/bboxes/bboxes/%s.txt"


# Merge config dictionary
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the options in b whenever they are also specified in a."""
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(
                    ("Type mismatch ({} vs. {}) " "for config key: {}").format(
                        type(b[k]), type(v), k
                    )
                )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(("Error under config key: {}".format(k)))
                raise
        else:
            b[k] = v


# load yaml file
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, "r", encoding="utf-8") as f:
        yaml_cfg = edict(yaml.load(f,Loader=Loader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_update(args):
    """Overwrite the hyperparameters in cfg."""
    # the path of model
    if args.weights is not None:
        cfg.CONST.weights = args.weights
    cfg.CONST.device = args.gpu_id
    if args.workdir is not None:
        cfg.DIR.out_path = args.workdir

    # set up folders for logs and checkpoints
    output_dir = os.path.join(
        cfg.DIR.out_path, "%s", datetime.datetime.now().isoformat()
    )
    cfg.DIR.checkpoints = output_dir % "checkpoints"
    cfg.DIR.logs = output_dir % "logs"
    return output_dir