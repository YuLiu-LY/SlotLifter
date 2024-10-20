# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from dataset.scannet_dataset import ScannetDataset
from dataset.uorf_dataset import MultiscenesDataset
from dataset.dtu_dataset import DTUDataset

datasets = {
    "scannet": ScannetDataset,
    "uorf": MultiscenesDataset,
    "dtu": DTUDataset,
}

def get_dataset(config):
    dataset = datasets[config.dataset]
    train_set = dataset(config, "train")
    val_set = dataset(config, "val")
    test_set = dataset(config, "test")
    return train_set, val_set, test_set

