import argparse
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as fun
import statistics

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    ##### Dataset #####
    # parser.add_argument(
    #     "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"]
    # )
    # parser.add_argument(
    #     "--dataset_root_dir",
    #     type=str,
    #     default=None,
    #     help="Root directory containing the dataset files. \
    #           If None, default values from scannet.py/sunrgbd.py are used",
    # )
    # parser.add_argument(
    #     "--meta_data_dir",
    #     type=str,
    #     default=None,
    #     help="Root directory containing the metadata files. \
    #           If None, default values from scannet.py/sunrgbd.py are used",
    # )
    # parser.add_argument("--dataset_num_workers", default=4, type=int)
    # parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    return parser

torch.manual_seed(1)

parser = make_args_parser()
args = parser.parse_args()
print(args)
from detr_br import build_3detr
model, output_processor = build_3detr(args)
# print(model)
model.to('cuda')

from lidar_br import KittiDetectionDataset
root = './kitti_object_vis/data/object/training'
label_path = './kitti_object_vis/data/object/training/label_2'

dataset = KittiDetectionDataset(root, label_path)
print(torch.stack([torch.FloatTensor(dataset[0][2])[..., :3]]*2).shape)

with torch.no_grad():
  output = model.run_encoder(
    torch.stack([torch.FloatTensor(dataset[0][2])[..., :3]]*2).to('cuda')
    )

print(output[1]['combined'][0].shape)
concatenated = torch.cat(output[1]['combined'], dim=2)
print(concatenated.shape)

fc_combine = nn.Linear(768, 768*2).to('cuda')

with torch.no_grad():
  concatenated = fun.relu(fc_combine(concatenated))
print(concatenated.shape)
max_pool = nn.MaxPool1d(1024, stride=1)
max_pooled = max_pool(concatenated.permute(0, 2, 1))
max_embedding = max_pooled.permute((0, 2, 1)).squeeze(1)
print(max_embedding.shape)

avg_pool = nn.AvgPool1d(1024, stride=1)
avg_pooled = avg_pool(concatenated.permute(0, 2, 1))
avg_embedding = avg_pooled.permute((0, 2, 1)).squeeze(1)
print(avg_embedding.shape)

global_embedding = torch.cat([max_embedding, avg_embedding], dim = 1)
print(global_embedding.shape)

fc_project = nn.Linear(768*2*2, 768*2*2).to('cuda')

with torch.no_grad():
  global_projected = fun.relu(fc_project(global_embedding))
print(global_projected.shape)
print(global_projected)