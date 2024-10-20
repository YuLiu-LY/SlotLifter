# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from dataset.scannet_dataset import ScannetDataset
from dataset.uorf_dataset import MultiscenesDataset, VisualDataset
from dataset.dtu_dataset import DTUDataset, VisDTUDataset

datasets = {
    "scannet": ScannetDataset,
    "uorf": MultiscenesDataset,
    "dtu": DTUDataset,
    "uorf_vis": VisualDataset,
    "dtu_vis": VisDTUDataset,
}

def get_dataset(config):
    if config.job_type == 'vis':
        dataset = datasets[config.dataset + "_vis"]
    else:
        dataset = datasets[config.dataset]
    train_set = dataset(config, "train")
    val_set = dataset(config, "val")
    test_set = dataset(config, "test")
    return train_set, val_set, test_set

