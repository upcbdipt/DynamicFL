#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : server.py
@Time    : 2022/07/30 16:08:31
@Desc    : 
"""
import network
from argparse import ArgumentParser
# FedAvg
from gfl.normal.core import strategy
from gfl.normal.core.server import FLClusterServer
from gfl.normal.core.job_manager import JobManager
from gfl.normal.utils.res_config import ResServerConfig
from gfl.normal.utils.utils import LoggerFactory
# Our Dynamic FL
from gfl.dynamic.core import strategy as DynamicStrategy
from gfl.dynamic.core.server import FLClusterServer as DynamicFLClusterServer
from gfl.dynamic.core.job_manager import JobManager as DynamicJobManager
from gfl.dynamic.utils.res_config import ResServerConfig as DynamicResServerConfig
from gfl.dynamic.utils.utils import LoggerFactory as DynamicLoggerFactory


parser = ArgumentParser("Dynamic Federated Learning Job Model Configuration !!! ")
parser.add_argument("-b", "--backbone", default="ghost_net", type=str, choices=["ghost_net", "resnet50", "resnet101"], help="The network for federated learning")
parser.add_argument("-cs", "--comm_step", default=50, type=int, help="The total aggregation communication step")
parser.add_argument("-alg", "--alg", default="DynamicFL", type=str, choices=["FedAvg", "DynamicFL"], help="The aggregation algorithm")
parser.add_argument("-ip", "--ip", default="127.0.0.1", type=str, help="The aggregation server url ip")
parser.add_argument("-port", "--port", default=9770, type=int, help="The aggregation server url port")


if __name__ == "__main__":

    args = parser.parse_args()
    # generate Federated Learning job
    func = getattr(network, args.backbone)
    model = func()
    if args.alg == "FedAvg":
        ResServerConfig.set_res_type("server")
        LoggerFactory.init_logger(ResServerConfig.log_file())
        job_manager = JobManager(ResServerConfig.JOB_PATH, ResServerConfig.BASE_MODEL_PATH)
        job = job_manager.generate_job(
            work_mode=strategy.WorkModeStrategy.WORKMODE_CLUSTER,
            fed_strategy=strategy.FederateStrategy.FED_AVG,
            epoch=args.comm_step,
            model=func
        )
        job_manager.submit_job(job, model)

        # Run Aggregator Server
        FLClusterServer(
            strategy.FederateStrategy.FED_AVG,
            args.ip,
            int(args.port),
            "/api/version",
            data=None,
            job_path=ResServerConfig.JOB_PATH,
            base_model_path=ResServerConfig.BASE_MODEL_PATH
        ).start()

    else:
        DynamicResServerConfig.set_res_type("server")
        DynamicLoggerFactory.init_logger(DynamicResServerConfig.log_file())
        job_manager = DynamicJobManager(DynamicResServerConfig.JOB_PATH, DynamicResServerConfig.BASE_MODEL_PATH)
        job = job_manager.generate_job(
            work_mode=DynamicStrategy.WorkModeStrategy.WORKMODE_CLUSTER,
            fed_strategy=DynamicStrategy.FederateStrategy.FED_AVG,
            epoch=args.comm_step,
            model=func
        )
        job_manager.submit_job(job, model)

        # Run Aggregator Server
        DynamicFLClusterServer(
            DynamicStrategy.FederateStrategy.FED_AVG,
            args.ip,
            int(args.port),
            "/api/version",
            data=None,
            job_path=DynamicResServerConfig.JOB_PATH,
            base_model_path=DynamicResServerConfig.BASE_MODEL_PATH
        ).start()
