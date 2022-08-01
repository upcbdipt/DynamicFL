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
import importlib
from gfl.normal.utils.utils import JobUtils, ModelUtils
from gfl.normal.entity.model import Model
from gfl.normal.utils.res_config import ResClientConfig


# JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
# BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")


class FLClient(object):
    def __init__(self, job_path, base_model_path, res_type):
        super(FLClient, self).__init__()
        self.job_path = job_path
        self.base_model_path = base_model_path
        self.res_type = res_type


    def get_remote_pfl_models(self, server_url=None):
        if server_url is None:
            return self._get_models_from_local()
        else:
            return self._get_models_from_remote(server_url)


    def _get_models_from_local(self):
        model_list = []
        JobUtils.get_job_from_remote(None, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)

        for job in job_list:
            model = self._get_model_from_job(job, None)
            pfl_model = Model()
            pfl_model.set_model(model)
            pfl_model.set_job_id(job.get_job_id())
            model_list.append(pfl_model)
        return model_list

    def _get_models_from_remote(self, server_url):
        model_list = []
        JobUtils.get_job_from_remote(server_url, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)
        for job in job_list:
            model = self._get_model_from_job(job, server_url)
            pfl_model = Model()
            pfl_model.set_model(model)
            pfl_model.set_job_id(job.get_job_id())
            model_list.append(pfl_model)
        return model_list

    def _get_model_from_job(self, job, server_url):
        job_id = job.get_job_id()

        init_model_path = os.path.join(self.base_model_path, "models_{}".format(job_id))
        if not os.path.exists(init_model_path):
            os.makedirs(init_model_path)
        init_model_filename = os.path.join(init_model_path, "init_model_{}.py".format(job_id))
        if server_url is not None and not os.path.exists(init_model_filename):
            ModelUtils.get_init_model_from_remote(server_url, init_model_path, job_id)

        module = importlib.import_module(
            f"res.{self.res_type}.models.models_{job_id}.init_model_{job_id}",
            "init_model_{}".format(job_id)
        )
        model_class = getattr(module, job.get_train_model_class_name())
        return model_class()




