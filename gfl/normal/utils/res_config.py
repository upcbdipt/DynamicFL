#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : res_config.py
@Time    : 2022/07/31 10:21:12
@Desc    : 
"""
import os


class ResServerConfig:
    RES_TYPE = "server"
    # logger
    LOG_DIR = os.path.join(os.path.abspath("."), "res", RES_TYPE)
    LOG_FILE = os.path.join(LOG_DIR, "log.txt")
    # resource 
    JOB_PATH = os.path.join(os.path.abspath("."), "res", RES_TYPE, "jobs_server")
    BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", RES_TYPE, "models")

    @staticmethod
    def log_file():
        if not os.path.exists(ResServerConfig.LOG_DIR):
            os.makedirs(ResServerConfig.LOG_DIR)
        return ResServerConfig.LOG_FILE

    @staticmethod
    def set_res_type(res_type):
        ResServerConfig.RES_TYPE = res_type
        ResServerConfig.LOG_DIR = os.path.join(os.path.abspath("."), "res", ResServerConfig.RES_TYPE)
        ResServerConfig.LOG_FILE = os.path.join(ResServerConfig.LOG_DIR, "log.txt")
        ResServerConfig.JOB_PATH = os.path.join(os.path.abspath("."), "res", ResServerConfig.RES_TYPE, "jobs_server")
        ResServerConfig.BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", ResServerConfig.RES_TYPE, "models")
    pass


class ResClientConfig:
    RES_TYPE = "n1"
    # logger
    LOG_DIR = os.path.join(os.path.abspath("."), "res", RES_TYPE)
    LOG_FILE = os.path.join(LOG_DIR, "log.txt")

    # resource 
    JOB_PATH = os.path.join(os.path.abspath("."), "res", RES_TYPE, "jobs_server")
    BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", RES_TYPE, "models")

    @staticmethod
    def log_file():
        if not os.path.exists(ResClientConfig.LOG_DIR):
            os.makedirs(ResClientConfig.LOG_DIR)
        return ResClientConfig.LOG_FILE

    @staticmethod
    def set_res_type(res_type):
        ResClientConfig.RES_TYPE = res_type
        ResClientConfig.LOG_DIR = os.path.join(os.path.abspath("."), "res", ResClientConfig.RES_TYPE)
        ResClientConfig.LOG_FILE = os.path.join(ResClientConfig.LOG_DIR, "log.txt")

        ResClientConfig.JOB_PATH = os.path.join(os.path.abspath("."), "res", ResClientConfig.RES_TYPE, "jobs_server")
        ResClientConfig.BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", ResClientConfig.RES_TYPE, "models")
    pass