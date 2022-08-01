#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2017, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : dynamic_server_config.py
@Time    : 2020/8/23 15:08
@Desc    :
"""
import time
import logging
import requests
import threading
from gfl.dynamic.utils.utils import LoggerFactory

logger = LoggerFactory.getLogger("DynamicServerConfig", logging.INFO)


class Waiter(object):
    def waiting(self):
        # 当前的时间是否为开启等待情况
        if DynamicServerConfig.get_start_waiting():
            # 当所有的节点都已上传模型，或者超过等待时间则设置模型可聚合
            if int(time.time()) >= DynamicServerConfig.get_waiting_end_time() or \
                    len(
                        DynamicServerConfig.aggregate_request_client_list
                    ) == len(
                            DynamicServerConfig.connected_client_id_list
                        ) + DynamicServerConfig.get_offline_client_count():
                DynamicServerConfig.set_ready_for_aggregate(True)
                # 生成所有需要进行模型下发的客户端的ip：port地址
                for client_id in DynamicServerConfig.aggregate_request_client_list:
                    if client_id in DynamicServerConfig.connected_client_id_ip_map:
                        DynamicServerConfig.dispatch_client_list.append(
                            DynamicServerConfig.connected_client_id_ip_map[client_id]
                        )
                # 有模型上传，即客户端收到了训练节点的网络模型
                if DynamicServerConfig.aggregate_client:
                    # 通知客户端即将下发模型，下一次训练需要加载新的模型
                    for client in DynamicServerConfig.dispatch_client_list:
                        client_url = "http://{}".format(client)
                        requests.post(
                            "/".join(
                                [client_url, "notify_dispatch", "dispatch"]
                            ), data=None
                        )
                elif DynamicServerConfig.none_aggregate_client and \
                        len(
                            DynamicServerConfig.none_aggregate_client
                        ) == len(DynamicServerConfig.connected_client_id_list):
                    # 通知客户端即将下发模型，下一次训练需要加载新的模型
                    for client in DynamicServerConfig.dispatch_client_list:
                        client_url = "http://{}".format(client)
                        requests.post(
                            "/".join(
                                [client_url, "notify_dispatch", "not dispatch"]
                            ), data=None
                        )
                    # 所有的客户端都没上传模型，此时重置模型融合的参数请求
                    DynamicServerConfig.clear_aggregate_config()
                    DynamicServerConfig.dispatch_client_list.clear()
                else:
                    pass
                # 模型融合标记设置为True则关闭等待标志
                DynamicServerConfig.set_start_waiting(False)
            else:
                pass
        pass
    pass


class DynamicServerConfig(object):
    _instance_lock = threading.Lock()
    _whether_first_request = True
    _default_time = 30
    _start_waiting = False
    _waiting_end_time = None
    client_training_time = dict()

    # 配置用于记录所有客户端的ID以及ID与IP的映射关系
    connected_client_id_list = list()
    connected_client_id_ip_map = dict()

    # 配置哪些节点需要融合以及哪些节点不需要融合，以及下发模型的节点列表
    aggregate_client = list()
    none_aggregate_client = list()
    aggregate_request_client_list = list()
    dispatch_client_list = list()

    _ready_for_aggregate = False
    # 用于记录在模型融合请求过程中完成训练的节点离线的数量
    _offline_client_count = 0

    @staticmethod
    def clear_aggregate_config():
        DynamicServerConfig.set_whether_first_request(True)
        DynamicServerConfig.aggregate_client.clear()
        DynamicServerConfig.none_aggregate_client.clear()
        DynamicServerConfig.aggregate_request_client_list.clear()
        DynamicServerConfig._ready_for_aggregate = False
        DynamicServerConfig.set_offline_client_count(0)
        pass

    @staticmethod
    def set_offline_client_count(offline_client_count):
        DynamicServerConfig._offline_client_count = offline_client_count

    @staticmethod
    def get_offline_client_count():
        return DynamicServerConfig._offline_client_count

    @staticmethod
    def set_start_waiting(start_waiting):
        DynamicServerConfig._start_waiting = start_waiting

    @staticmethod
    def get_start_waiting():
        return DynamicServerConfig._start_waiting

    @staticmethod
    def set_ready_for_aggregate(ready_for_aggregate):
        DynamicServerConfig._ready_for_aggregate = ready_for_aggregate

    @staticmethod
    def get_ready_for_aggregate():
        return DynamicServerConfig._ready_for_aggregate

    @staticmethod
    def set_whether_first_request(whether_first_request):
        DynamicServerConfig._whether_first_request = whether_first_request
        pass

    @staticmethod
    def get_whether_first_request():
        return DynamicServerConfig._whether_first_request
        pass

    @staticmethod
    def set_waiting_end_time(start_time):
        DynamicServerConfig._waiting_end_time = start_time + DynamicServerConfig.calculate_wait_time()

    @staticmethod
    def get_waiting_end_time():
        return DynamicServerConfig._waiting_end_time

    @staticmethod
    def calculate_wait_time():
        import numpy as np

        def dict2array(dicts):
            arrays = []
            for key in dicts.keys():
                arrays.append(int(dicts[key]))
            return arrays

        # 中位数策略求等待时间
        def calculate_median(arrays):
            # 排序
            arrays.sort()
            length = len(arrays)
            if length % 2 == 0:
                return int((arrays[length // 2] + arrays[(length // 2) - 1]) / 2)
            else:
                return int(arrays[length // 2])

        # 正态分布的3sigma计算等待时间
        def calculate_3_sigma(arrays):
            mean = np.mean(arrays)
            var = np.var(arrays)
            return int(mean + 3 * var)

        # 训练时间列表非空，并且训练时间列表长度与已连接到中心节点的列表长度一致
        if DynamicServerConfig.client_training_time and \
                len(DynamicServerConfig.client_training_time) == len(DynamicServerConfig.connected_client_id_list):
            arr = dict2array(DynamicServerConfig.client_training_time)
            # 等待时间设置策略
            waiting_time = calculate_median(arr)
            logger.info("new waiting time: {}".format(waiting_time))
            return waiting_time
        else:
            logger.info("default waiting time: {}".format(DynamicServerConfig._default_time))
            return DynamicServerConfig._default_time

    def __init__(self, *args, **kwargs):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with DynamicServerConfig._instance_lock:
                if not hasattr(cls, '_instance'):
                    DynamicServerConfig._instance = super().__new__(cls)
        return DynamicServerConfig._instance
