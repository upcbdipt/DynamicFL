# Copyright (c) 2019 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging
from werkzeug.serving import run_simple
from flask import Flask, send_from_directory, request
from gfl.dynamic.core.job_manager import JobManager
from gfl.dynamic.utils.utils import JobEncoder, return_data_decorator, LoggerFactory, CyclicTimer
from gfl.dynamic.entity.runtime_config import CONNECTED_TRAINER_LIST
from gfl.dynamic.entity.dynamic_server_config import *
from gfl.dynamic.utils.res_config import ResServerConfig


API_VERSION = "/api/v1"
# JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
# BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")

app = Flask(__name__)
logger = LoggerFactory.getLogger("communicate_server", logging.INFO)

for handler in LoggerFactory.getHandlers():
    logging.root.addHandler(handler)


@app.route("/test/<name>")
@return_data_decorator
def test_flask_server(name):
    return name, 200


@app.route("/register/<ip>/<port>/<client_id>", methods=['POST'], endpoint='register_trainer')
@return_data_decorator
def register_trainer(ip, port, client_id):
    trainer_host = ip + ":" + port
    if trainer_host not in CONNECTED_TRAINER_LIST:
        job_list = JobManager.get_job_list(ResServerConfig.JOB_PATH)
        for job in job_list:
            job_model_client_dir = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job.get_job_id()),
                                                "models_{}".format(client_id))
            if not os.path.exists(job_model_client_dir):
                os.makedirs(job_model_client_dir)
        CONNECTED_TRAINER_LIST.append(trainer_host)
        """
        自定义的策略需要的配置
        """
        # 记录所有的客户端的ID
        DynamicServerConfig.connected_client_id_list.append(client_id)
        # 记录所有节点ID与IP的映射，如果已有则更新
        DynamicServerConfig.connected_client_id_ip_map[client_id] = trainer_host
        return 'register_success', 200
    else:
        return 'already connected', 201


@app.route("/offline/<ip>/<port>/<client_id>", methods=['PUT'], endpoint='offline')
@return_data_decorator
def offline(ip, port, client_id):
    trainer_host = ip + ":" + port
    if trainer_host in CONNECTED_TRAINER_LIST:
        CONNECTED_TRAINER_LIST.remove(trainer_host)
        # 删除记录的客户端的ID
        DynamicServerConfig.connected_client_id_list.remove(client_id)
        # 如果该删除的客户端已经发送模型融合请求，则需要记录离线客户端的数量
        if client_id in DynamicServerConfig.aggregate_request_client_list:
            DynamicServerConfig.set_offline_client_count(
                DynamicServerConfig.get_offline_client_count() + 1
            )
        else:
            # 删除指定的字典中的值
            del DynamicServerConfig.connected_client_id_ip_map[client_id]
        return 'offline success', 200
    return 'already offline', 201


@app.route("/jobs", methods=['GET'], endpoint='acquire_job_list') # type: ignore
@return_data_decorator
def acquire_job_list():
    job_str_list = []
    job_list = JobManager.get_job_list(ResServerConfig.JOB_PATH)
    for job in job_list:
        job_str = json.dumps(job, cls=JobEncoder)
        job_str_list.append(job_str)
    return job_str_list, 200


@app.route("/modelpars/<job_id>", methods=['GET'], endpoint='acquire_init_model_pars')
def acquire_init_model_pars(job_id):
    init_model_pars_dir = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id))
    return send_from_directory(init_model_pars_dir, "init_model_pars_{}".format(job_id), as_attachment=True)


@app.route("/init_model/<job_id>", methods=['GET'], endpoint='acquire_init_model')
def acquire_init_model(job_id):
    init_model_path = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id))
    return send_from_directory(init_model_path, "init_model_{}.py".format(job_id), as_attachment=True)


@app.route("/modelpars/<client_id>/<job_id>", methods=['POST'], endpoint='submit_model_parameter')
@return_data_decorator
def submit_model_parameter(client_id, job_id):
    tmp_parameter_file = request.files['tmp_parameter_file']
    model_pars_dir = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id), "models_{}".format(client_id))
    if not os.path.exists(model_pars_dir):
        os.makedirs(model_pars_dir)
    # 获取最新的fed_step，如果模型融合文件路径不存在，则创建该路径，并表示刚开始第一次融合
    aggregate_model_dir = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id), "tmp_aggregate_pars")
    if not os.path.exists(aggregate_model_dir):
        os.makedirs(aggregate_model_dir)
        fed_step = 1
    else:
        fed_step = len(os.listdir(aggregate_model_dir)) + 1
    model_pars_path = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id), "models_{}".format(client_id),
                                   "tmp_parameters_{}".format(fed_step))
    with open(model_pars_path, "wb") as f:
        for line in tmp_parameter_file.readlines():
            f.write(line)

    return 'submit_success', 200


@app.route("/otherparameters/<job_id>/<client_id>/<fed_step>", methods=['GET'], endpoint='get_other_parameters')
def get_other_parameters(job_id, client_id, fed_step):
    tmp_parameter_dir = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id), "models_{}".format(client_id))
    tmp_parameter_path = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id), "models_{}".format(client_id),
                                      "tmp_parameters_{}".format(fed_step))

    if not os.path.exists(tmp_parameter_path):
        return 'file not prepared', 201

    return send_from_directory(tmp_parameter_dir, "tmp_parameters_{}".format(fed_step), as_attachment=True), 202


@app.route("/otherclients/<job_id>", methods=['GET'], endpoint='get_connected_clients') # type: ignore
@return_data_decorator
def get_connected_clients(job_id):
    connected_clients_id = []
    job_model_path = os.path.join(ResServerConfig.BASE_MODEL_PATH, "models_{}".format(job_id))
    for model_dir in os.listdir(job_model_path):
        if model_dir.find("models_") != -1:
            connected_clients_id.append(int(model_dir.split("_")[-1]))

    return connected_clients_id, 200


@app.route("/aggregatepars", methods=['GET'], endpoint='get_aggregate_parameter')
@return_data_decorator
def get_aggregate_parameter():
    return ''

"""
自定义的api接口为了实现
    - 模型动态融合
    - 基于机器能力强弱的动态模型融合
"""
# 配置哪些节点需要进行模型的融合，哪些节点不需要融合
@app.route("/aggregate_request/<client_id>/<types>", methods=['POST'], endpoint='submit_model_client')
@return_data_decorator
def submit_model_client(client_id, types):
    # 第一个请求模型融合需要记录融合的开始时间以及开启等待
    if DynamicServerConfig.get_whether_first_request():
        DynamicServerConfig.set_whether_first_request(False)
        DynamicServerConfig.set_start_waiting(True)
        DynamicServerConfig.set_waiting_end_time(int(time.time()))

    # 是否超时，即当前时间超过等待截止时间
    if int(time.time()) <= DynamicServerConfig.get_waiting_end_time():  # type: ignore
        if types == "aggregate" and client_id not in DynamicServerConfig.aggregate_client:
            # 记录已经上传模型的节点列表，以及发送融合请求所有的客户端的列表
            DynamicServerConfig.aggregate_client.append(client_id)
            DynamicServerConfig.aggregate_request_client_list.append(client_id)
            return "upload", 200
        elif types == "none_aggregate" and client_id not in DynamicServerConfig.none_aggregate_client:
            # 记录发送上传请求但是不上传模型的节点列表，以及发送融合请求所有的客户端的列表
            DynamicServerConfig.none_aggregate_client.append(client_id)
            DynamicServerConfig.aggregate_request_client_list.append(client_id)
            return "not_upload", 200
        else:
            return "client_recorded", 201
    else:
        return "timeout", 200
        pass


@return_data_decorator
@app.route("/notify_capacity/<client_id>/<train_time>", methods=['POST'])
def notify_upload(client_id, train_time):
    DynamicServerConfig.client_training_time[client_id] = train_time
    return "ok", 200


def start_communicate_server(api_version, ip, port):
    # 配置动态融合的参数监听器
    waiter = Waiter()
    t = CyclicTimer(2, waiter.waiting)
    t.start()
    # 启动Flask任务
    app.url_map.strict_slashes = False
    run_simple(hostname=ip, port=port, application=app, threaded=True)
    logger.info("galaxy learning server started")
