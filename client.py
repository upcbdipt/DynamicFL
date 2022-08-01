#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : client.py
@Time    : 2022/07/30 16:09:10
@Desc    : 
"""
import torch
from torchvision import datasets, transforms
from argparse import ArgumentParser

# FedAvg
from gfl.normal.core import strategy
from gfl.normal.core.client import FLClient
from gfl.normal.core.trainer_controller import TrainerController
from gfl.normal.utils.res_config import ResClientConfig
from gfl.normal.utils.utils import LoggerFactory
# Our Dynamic FL
from gfl.dynamic.core import strategy as DynamicStrategy
from gfl.dynamic.core.client import FLClient as DynamicFLClient
from gfl.dynamic.core.trainer_controller import TrainerController as DynamicTrainerController
from gfl.dynamic.utils.res_config import ResClientConfig as DynamicResClientConfig
from gfl.dynamic.utils.utils import LoggerFactory as DynamicLoggerFactory


parser = ArgumentParser("FL arguments")
parser.add_argument("-alg", "--alg", default="DynamicFL", type=str, choices=["FedAvg", "DynamicFL"])
parser.add_argument("-d", "--datadir", default="data/preprocess/N1", type=str, help="The dataset root dir")

parser.add_argument("-c", "--client", default=1, type=int, help="The DynamicFL client id")
parser.add_argument("-p", "--port", default=9080, type=int, help="The DynamicFL client id")
parser.add_argument("-ip", "--ip", default="127.0.0.1", type=str, help="The client ip for training")
parser.add_argument("-su", "--server_url", default="http://127.0.0.1:9770", type=str, help="The aggregation server url")

parser.add_argument("-ls", "--ls", default=3, type=int, help="The optimizer local epoch for clients training in every communication step")
parser.add_argument("-lr", "--lr", default=0.001, type=float, help="The optimizer learning rate for training")
parser.add_argument("-mo", "--momentum", default=0.9, type=float, help="The optimizer momentum for training")
parser.add_argument("-bs", "--bs", default=8, type=int, help="The optimizer batch size for training")

args = parser.parse_args()


if __name__ == "__main__":
    args = parser.parse_args()
    # loading data
    train_data = datasets.ImageFolder(
        f"{args.datadir}/train",
        transform=transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            # 将 (0, 1)转换成(-1, 1)，即((0, 1)- 0.5) / 0.5: (-1, 1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    test_data = datasets.ImageFolder(
        f"{args.datadir}/test",
        transform=transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            # 将 (0, 1)转换成(-1, 1)，即((0, 1)- 0.5) / 0.5: (-1, 1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    # run client
    if args.alg == "FedAvg":
        ResClientConfig.set_res_type(f"n{args.client}")
        LoggerFactory.init_logger(ResClientConfig.log_file())
        client = FLClient(ResClientConfig.JOB_PATH, ResClientConfig.BASE_MODEL_PATH, ResClientConfig.RES_TYPE)
        gfl_models = client.get_remote_pfl_models(args.server_url)

        for gfl_model in gfl_models:
            optimizer = torch.optim.SGD(gfl_model.get_model().parameters(), lr=args.lr, momentum=args.momentum)
            train_strategy = strategy.TrainStrategy(optimizer=optimizer, batch_size=args.bs, loss_function=strategy.LossStrategy.NLL_LOSS)
            gfl_model.set_train_strategy(train_strategy)

        TrainerController(
            work_mode=strategy.WorkModeStrategy.WORKMODE_CLUSTER, models=gfl_models,
            train_data=train_data, valid_data=test_data,
            local_epoch=args.ls, client_id=args.client, client_ip=args.ip,
            client_port=args.port, server_url=args.server_url,
            curve=False, concurrent_num=3,
            job_path=ResClientConfig.JOB_PATH, base_model_path=ResClientConfig.BASE_MODEL_PATH
        ).start()
    else:
        DynamicResClientConfig.set_res_type(f"n{args.client}")
        DynamicLoggerFactory.init_logger(DynamicResClientConfig.log_file())
        client = DynamicFLClient(DynamicResClientConfig.JOB_PATH, DynamicResClientConfig.BASE_MODEL_PATH, DynamicResClientConfig.RES_TYPE)
        gfl_models = client.get_remote_pfl_models(args.server_url)

        for gfl_model in gfl_models:
            optimizer = torch.optim.SGD(gfl_model.get_model().parameters(), lr=args.lr, momentum=args.momentum)
            train_strategy = DynamicStrategy.TrainStrategy(optimizer=optimizer, batch_size=args.bs, loss_function=DynamicStrategy.LossStrategy.NLL_LOSS)
            gfl_model.set_train_strategy(train_strategy)

        DynamicTrainerController(
            work_mode=DynamicStrategy.WorkModeStrategy.WORKMODE_CLUSTER, models=gfl_models,
            train_data=train_data, valid_data=test_data,
            local_epoch=args.ls, client_id=args.client, client_ip=args.ip, client_port=args.port,
            server_url=args.server_url, curve=False, concurrent_num=3,
            job_path=DynamicResClientConfig.JOB_PATH, base_model_path=DynamicResClientConfig.BASE_MODEL_PATH
        ).start()
