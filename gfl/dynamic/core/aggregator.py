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
import torch
import logging
import requests
import portalocker
from concurrent.futures import ThreadPoolExecutor
from gfl.dynamic.utils.utils import LoggerFactory
from gfl.dynamic.core.job_manager import JobManager
from gfl.dynamic.core.strategy import WorkModeStrategy
from gfl.dynamic.entity.runtime_config import WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST
from gfl.dynamic.entity.dynamic_server_config import *

LOCAL_AGGREGATE_FILE = os.path.join("tmp_aggregate_pars", "avg_pars")


class Aggregator(object):
    """
    Aggregator is responsible for aggregating model parameters
    """

    def __init__(self, work_mode, job_path, base_model_path, concurrent_num=5):
        self.job_path = job_path
        self.base_model_path = base_model_path
        self.aggregate_executor_pool = ThreadPoolExecutor(concurrent_num)
        self.work_mode = work_mode

    def load_model_pars(self, job_model_pars_path, fed_step):
        """
        :param job_model_pars_path: 模型文件的根路径: res/models/models_{job_id}/
        :param fed_step: 初始化时该值为 None
        :return:
        """
        if not DynamicServerConfig.get_ready_for_aggregate():
            return None, 0
        fed_step = 0 if fed_step is None else fed_step
        job_model_pars = []
        last_model_par_file_num = 0
        for f in os.listdir(job_model_pars_path):
            client_id = f.split("_")[-1] if len(f.split("_")) == 2 else -1
            """当前节点（即节点对应的文件夹的模型文件[不需要上传，因而存在该文件]）不需要进行模型融合"""
            if client_id in DynamicServerConfig.none_aggregate_client:
                continue
            # 当前节点（即节点对应的文件夹的模型文件）需要进行模型的融合
            elif f.find("models_") != -1 and client_id in DynamicServerConfig.aggregate_client:
                one_model_par_path = os.path.join(job_model_pars_path, f)
                one_model_par_files = os.listdir(one_model_par_path)
                if one_model_par_files and len(one_model_par_files) != 0:
                    last_model_par_file_num = self._find_last_model_file_num(one_model_par_files)
                    if last_model_par_file_num > fed_step:
                        # one_model_par_files[-1]为取出文件列表中的最后一个文件名, 若此时文件排列为倒叙,则会获取第一次的模型
                        # 即[tmp_parameters_2, tmp_parameters_1], 则会一直获取到tmp_parameters_1文件
                        # tmp_model_path = os.path.join(one_model_par_path, one_model_par_files[-1])
                        # 使用下列语句代替上一句语句则可以一直获取到最新的模型
                        tmp_model_path = os.path.join(one_model_par_path,
                                                      "tmp_parameters_{}".format(last_model_par_file_num))

                        # 该文件不存在, 则返回None
                        if not os.path.exists(tmp_model_path):
                            return None, 0
                        # 获取文件锁, 判断该文件是否正在写入
                        with open(tmp_model_path) as file:
                            portalocker.lock(file, portalocker.LOCK_EX)
                        try:
                            model_par = torch.load(tmp_model_path)
                        except EOFError:
                            return None, 0
                        except RuntimeError:
                            return None, 0
                        job_model_pars.append(model_par)
                    else:
                        return None, 0
                else:
                    return None, 0
        return job_model_pars, last_model_par_file_num

    def _find_last_model_file_num(self, files):
        """
        :param files:
        :return:
        """
        last_num = 0
        for file in files:
            file_num = int(file.split("_")[-1])
            last_num = file_num if last_num < file_num else last_num
        return last_num


class FedAvgAggregator(Aggregator):
    """
    FedAvgAggregator is responsible for aggregating model parameters by using FedAvg Algorithm
    """

    def __init__(self, work_mode, job_path, base_model_path):
        super(FedAvgAggregator, self).__init__(work_mode, job_path, base_model_path)
        self.fed_step = {}
        self.logger = LoggerFactory.getLogger("FedAvgAggregator", logging.INFO)

    def aggregate(self):
        """
        :return:
        """
        job_list = JobManager.get_job_list(self.job_path)
        WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST.clear()
        for job in job_list:
            # 获取训练节点上传的模型参数以及融合次数
            job_model_pars, fed_step = self.load_model_pars(
                os.path.join(self.base_model_path, "models_{}".format(job.get_job_id())),
                self.fed_step.get(job.get_job_id())
            )
            job_fed_step = 0 if self.fed_step.get(job.get_job_id()) is None else self.fed_step.get(job.get_job_id())
            if job_fed_step != fed_step and job_model_pars is not None and job_model_pars:
                """开始融合模型，因而需要重置上一次模型融合标记变量"""
                DynamicServerConfig.clear_aggregate_config()
                self.logger.info("Aggregating......")
                self._exec(job_model_pars, self.base_model_path, job.get_job_id(), fed_step)
                self.fed_step[job.get_job_id()] = fed_step
                WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST.append(job.get_job_id())
                if job.get_epoch() <= self.fed_step[job.get_job_id()]:
                    self._save_final_model_pars(job.get_job_id(),
                                                os.path.join(self.base_model_path,
                                                             "models_{}".format(job.get_job_id()),
                                                             "tmp_aggregate_pars"),
                                                self.fed_step[job.get_job_id()])
                if self.work_mode == WorkModeStrategy.WORKMODE_CLUSTER:
                    # 这里会将模型通下发每一个节点进行训练
                    # self._broadcast(WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST, CONNECTED_TRAINER_LIST,
                    #                 self.base_model_path)
                    """这里修改为只对当前等待时间范围内的节点进行下发模型与训练"""
                    self._broadcast(WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST, DynamicServerConfig.dispatch_client_list,
                                    self.base_model_path)
                    DynamicServerConfig.dispatch_client_list.clear()

    def _exec(self, job_model_pars, base_model_path, job_id, fed_step):
        avg_model_par = job_model_pars[0]
        for key in avg_model_par.keys():
            for i in range(1, len(job_model_pars)):
                avg_model_par[key] += job_model_pars[i][key]
            avg_model_par[key] = torch.div(avg_model_par[key], 1.0*len(job_model_pars))
        tmp_aggregate_dir = os.path.join(base_model_path, "models_{}".format(job_id), "tmp_aggregate_pars")
        tmp_aggregate_path = os.path.join(base_model_path, "models_{}".format(job_id),
                                          "{}_{}".format(LOCAL_AGGREGATE_FILE, fed_step))
        if not os.path.exists(tmp_aggregate_dir):
            os.makedirs(tmp_aggregate_dir)
        torch.save(avg_model_par, tmp_aggregate_path)

        self.logger.info("job: {} the {}th round parameters aggregated successfully!".format(job_id, fed_step))

    def _broadcast(self, job_id_list, connected_client_list, base_model_path):
        """

        :param job_id_list:
        :param connected_client_list:
        :param base_model_path:
        :return:
        """
        self.logger.info("connected client list: {}".format(connected_client_list))
        for client in connected_client_list:
            aggregated_files = self._prepare_upload_aggregate_file(job_id_list, base_model_path)
            client_url = "http://{}".format(client)
            requests.post("/".join([client_url, "aggregatepars"]), data=None, files=aggregated_files)

    def _prepare_upload_aggregate_file(self, job_id_list, base_model_path):
        """

        :param job_id_list:
        :param base_model_path:
        :return:
        """
        fed_step = 0
        aggregated_files = {}
        for job_id in job_id_list:
            tmp_aggregate_dir = os.path.join(base_model_path, "models_{}".format(job_id), "tmp_aggregate_pars")
            fed_step = self._find_last_model_file_num(os.listdir(tmp_aggregate_dir))
            send_aggregate_filename = "tmp_aggregate_{}_{}".format(job_id, fed_step)
            tmp_aggregate_path = os.path.join(tmp_aggregate_dir, "avg_pars_{}".format(fed_step))
            aggregated_files[send_aggregate_filename] = (send_aggregate_filename, open(tmp_aggregate_path, "rb"))
        return aggregated_files

    def _save_final_model_pars(self, job_id, tmp_aggregate_dir, fed_step):
        """

        :param job_id:
        :param tmp_aggregate_dir:
        :param fed_step:
        :return:
        """
        job_model_dir = os.path.join(self.base_model_path, "models_{}".format(job_id))
        final_model_pars_path = os.path.join(os.path.abspath("."), "final_model_pars_{}".format(job_id))
        last_aggregate_file = os.path.join(tmp_aggregate_dir, "avg_pars_{}".format(fed_step))
        with open(final_model_pars_path, "wb") as final_f:
            with open(last_aggregate_file, "rb") as f:
                for line in f.readlines():
                    final_f.write(line)

        self.logger.info("job {} save final aggregated parameters successfully!".format(job_id))


class DistillationAggregator(Aggregator):
    """
    Model distillation does not require a centralized server that we don't need to provide a Distillation aggregator
    """
    def __init__(self, work_mode, job_path, base_model_path):
        super(DistillationAggregator, self).__init__(work_mode, job_path, base_model_path)
        self.fed_step = {}

    def aggregate(self):
        pass
