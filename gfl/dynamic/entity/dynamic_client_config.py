#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2017, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : dynamic_client_config.py
@Time    : 2020/9/5 下午3:08
@Desc    : 
"""
import threading


class DynamicClientConfig(object):
    _instance_lock = threading.Lock()
    _need_load_model = True
    _server_is_dispatch = None

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def set_need_load_model(need_load_model):
        DynamicClientConfig._need_load_model = need_load_model
        pass

    @staticmethod
    def get_need_load_model():
        return DynamicClientConfig._need_load_model
        pass

    @staticmethod
    def set_server_is_dispatch(server_is_dispatch):
        DynamicClientConfig._server_is_dispatch = server_is_dispatch

    @staticmethod
    def get_server_is_dispatch():
        return DynamicClientConfig._server_is_dispatch

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with DynamicClientConfig._instance_lock:
                if not hasattr(cls, '_instance'):
                    DynamicClientConfig._instance = super().__new__(cls)
        return DynamicClientConfig._instance
