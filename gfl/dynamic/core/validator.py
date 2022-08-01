#!/usr/bin/env python 
# -*- encoding: utf-8 -*- 
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2017, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : validator.py 
@Time    : 2020/8/15 22:30 
@Desc    : 
"""
import os
import torch
import logging
import importlib
import portalocker
from gfl.dynamic.core.job_manager import JobManager
from gfl.dynamic.utils.utils import LoggerFactory


class Validator(object):
    def __init__(self, job_path, model_path, valid_data, valid_batch_size=16):
        self.fed_step = 0
        self.job_path = job_path
        self.base_model_path = model_path
        self.valid_data = valid_data
        self.logger = LoggerFactory.getLogger("Validator", logging.INFO)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.job_list = JobManager.get_job_list(self.job_path)
        self.dataloader = torch.utils.data.DataLoader(
                self.valid_data,
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True
        )

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

    def _get_new_model_par(self, job_model_pars_path):
        if not os.path.exists(job_model_pars_path):
            return None, 0
        model_par_files = os.listdir(job_model_pars_path)
        if model_par_files and len(model_par_files) != 0:
            last_model_par_file_num = self._find_last_model_file_num(model_par_files)
            # aggregate_file = os.path.join(job_model_pars_path, model_par_files[-1])
            aggregate_file = os.path.join(job_model_pars_path, "avg_pars_{}".format(last_model_par_file_num))
            with open(aggregate_file) as f:
                portalocker.lock(f, portalocker.LOCK_EX)
            try:
                model_par = torch.load(aggregate_file)
            except EOFError:
                return None, 0
            except RuntimeError:
                return None, 0
            return model_par, last_model_par_file_num
        else:
            return None, 0

    def _get_model_from_job(self, job):
        job_id = job.get_job_id()
        init_model_path = os.path.join(self.base_model_path, "models_{}".format(job_id))
        if not os.path.exists(init_model_path):
            os.makedirs(init_model_path)

        module = importlib.import_module("res.models.models_{}.init_model_{}".format(job_id, job_id),
                                         "init_model_{}".format(job_id))
        model_class = getattr(module, job.get_train_model_class_name())
        return model_class()

    def validate(self):
        for job in self.job_list:
            model = self._get_model_from_job(job)
            model = model.to(self.device)
            params, fed_step = self._get_new_model_par(os.path.join(
                self.base_model_path, "models_{}".format(job.get_job_id()), "tmp_aggregate_pars")
            )
            if params and self.fed_step != fed_step:
                self.fed_step = fed_step
                model.load_state_dict(params)
                model.eval()
                with torch.no_grad():
                    acc = 0
                    for idx, (valid_data, valid_target) in enumerate(self.dataloader):
                        valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)
                        pred = model(valid_data)
                        acc += torch.eq(pred.argmax(dim=1), valid_target).sum().float().item()
                    accuracy = acc / len(self.dataloader.dataset)
                self.logger.info("fed_step: {}, acc: {}".format(fed_step, accuracy))
            else:
                pass
        pass
