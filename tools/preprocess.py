#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : preprocess.py
@Time    : 2022/07/29 20:31:40
@Desc    : 
"""
import os
import shutil
import numpy as np
from itertools import accumulate
from argparse import ArgumentParser
from typing import Sequence


__CLASSES = ["COVID-19", "NORMAL", "Viral Pneumonia"]

__DATA_CONFIG = {
    1: {
        "CT":   [[281, 319], [0, 0], [0, 0]],
        "Xray": [[0, 0, 0], [72, 380, 448], [106, 548, 646]]
    }, 
    2: {
        "CT":   [[140, 160], [0, 0], [0, 0]],
        "Xray": [[24, 127, 149], [72, 380, 448], [106, 548, 646]]
    },
    3: {
        "CT":   [[94, 106], [0, 0], [0, 0]],
        "Xray": [[32, 169, 199], [72, 380, 448], [106, 548, 646]]
    },
    4: {
        "CT":   [[70, 80], [0, 0], [0, 0]],
        "Xray": [[36, 190, 224], [72, 380, 448], [106, 548, 646]]
    },
    5: {
        "CT":   [[94, 106], [94, 106], [0, 0]],
        "Xray": [[32, 169, 199], [57, 295, 348], [106, 548, 646]]
    },
    6: {
        "CT":   [[94, 106], [94, 106], [94, 106]],
        "Xray": [[32, 169, 199], [57, 295, 348], [89, 464, 547]]
    },
}

parser = ArgumentParser("The Dynamic Federated Aggregation Data Preprocessor.")
parser.add_argument("-dn", "--dataset_idx", default=1, type=int, choices=range(1, 7), help="The serveral datasets refers in paper")
parser.add_argument("-src", "--src", default="data/origin", type=str, help="The origin datasets root dir")
parser.add_argument("-dst", "--dst", default="data/preprocess", type=str, help="The preprocessed datasets save dir")
parser.add_argument("-m", "--merge", default=False, type=bool, choices=[True, False], help="Whether merge the same classes in the CT and Xray")


def print_ratios_in_nodes(dataset_idx: int):
    """
    print node data config
    :param int dataset_idx: data idx in data config
    :raises Exception:
    """
    data_config = __DATA_CONFIG.get(dataset_idx, {})
    CTs = data_config.get("CT", [])
    Xrays = data_config.get("Xray", [])
    if len(CTs) == len(Xrays):
        CT_Ns = [sum(lst) for lst in CTs]
        Xray_Ns = [sum(lst) for lst in Xrays]
        info = [ f"[N{i+1}] {c}/{x}" for i, (c, x) in enumerate(zip(CT_Ns, Xray_Ns)) ]
        print(", ".join(info))
    else:
        raise Exception("Input two arrays not equal")


def preprocess(dataset_idx, src: str, dst: str, merge: bool = False):
    """
    preprocess data
    :param _type_ dataset_idx: data idx in data config
    :param str src: data root dir
    :param str dst: preprocessed data save dir
    :param bool merge: if CT or Xrays are merged in the samge class, defaults to False
    """
    data_config = __DATA_CONFIG.get(dataset_idx, {})
    CTs = data_config.get("CT", [])
    Xrays = data_config.get("Xray", [])
    if merge:
        partition_dataset(CTs, f"{src}/train/CT", dst, __CLASSES[0:-1], "")
        partition_dataset(Xrays, f"{src}/train/Xray", dst, __CLASSES, "")
    else:
        partition_dataset(CTs, f"{src}/train/CT", dst, __CLASSES[0:-1], "CT_")
        partition_dataset(Xrays, f"{src}/train/Xray", dst, __CLASSES, "Xray_")
    
    # generate test dataset 
    for nidx in range(len(CTs)):
        node_dir = os.path.join(dst, f"N{nidx+1}")
        root = os.path.join(src, "test")
        if merge:
            for prefix in os.listdir(root):
                for class_name in os.listdir(os.path.join(root, prefix)):
                    shutil.rmtree(os.path.join(node_dir, "test", f"{prefix}_{class_name}"))
                    shutil.copytree(
                        os.path.join(root, prefix, class_name), 
                        os.path.join(node_dir, "test", f"{class_name}")
                    )
            pass
        else:
            for prefix in os.listdir(root):
                for class_name in os.listdir(os.path.join(root, prefix)):
                    shutil.rmtree(os.path.join(node_dir, "test", f"{prefix}_{class_name}"))
                    shutil.copytree(
                        os.path.join(root, prefix, class_name), 
                        os.path.join(node_dir, "test", f"{prefix}_{class_name}")
                    )
                    pass
                pass
            


def partition_dataset(nums: Sequence[Sequence[int]], src: str, dst: str, classes: Sequence[str], prefix: str):
    """
    partition origin dataset for dynamic federate aggregation
    :param Sequence[Sequence[int]] nums: data config per nodes
    :param str src: image root dir
    :param str dst: partitioned image save dir
    :param Sequence[str] classes: classes list
    :param str prefix: if CT or Xrays data can be merged
    :raises Exception: 
    """
    if len(nums) == 0 or len(classes) != len(nums[0]):
        raise Exception("Input classes must equals to per num length")
    for class_idx, class_name in enumerate(classes):
        # generate every node datasets start & end index in origin datasets
        endpoints = list(accumulate(np.array(nums)[:, class_idx], initial=0))
        # read image file names
        raw_dir = os.path.join(src, class_name)
        img_lst = os.listdir(raw_dir)
        # forloop endpoints
        for node_idx, (s, e) in enumerate(zip(endpoints[0:-1], endpoints[1:])):
             # generate node dir
            node_dir = os.path.join(dst, f"N{node_idx + 1}", "train")
            if not os.path.exists(node_dir):
                os.makedirs(node_dir)
            # generate partition image path
            class_dir = os.path.join(node_dir, f"{prefix}{class_name}")
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            copy(raw_dir, class_dir, img_lst[s:e])
    pass


def copy(src: str, dst: str, file_lst: Sequence[str]):
    """
    copy file list for src dir to dst dir
    :param str src: root image dir
    :param str dst: dst image dir
    :param Sequence[str] file_lst: file name list
    """
    for file in file_lst:
        src_file = os.path.join(src, file)
        dst_file = os.path.join(dst, file)
        shutil.copy(src_file, dst_file)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_idx = args.dataset_idx
    print_ratios_in_nodes(dataset_idx)
    preprocess(dataset_idx, args.src, args.dst, args.merge)
