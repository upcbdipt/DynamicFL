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
import os, logging, portalocker
from flask import Flask, request
from werkzeug.serving import run_simple
from gfl.dynamic.utils.utils import return_data_decorator, LoggerFactory
from gfl.dynamic.entity.dynamic_client_config import DynamicClientConfig
from gfl.dynamic.utils.res_config import ResClientConfig

# BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")

app = Flask(__name__)
logger = LoggerFactory.getLogger(__name__, logging.INFO)
for handler in LoggerFactory.getHandlers():
    logging.root.addHandler(handler)


@return_data_decorator
@app.route("/", methods=['GET'])
def test_client():
    return "Hello gfl client", 200


@return_data_decorator
@app.route("/aggregatepars", methods=['POST'])
def submit_aggregate_pars():
    # 设置模型需要加载的标志
    DynamicClientConfig.set_need_load_model(True)

    logger.info("receive aggregate files")
    recv_aggregate_files = request.files
    for filename in recv_aggregate_files:
        job_id = filename.split("_")[-2]
        fed_step = filename.split("_")[-1]
        tmp_aggregate_file = recv_aggregate_files[filename]
        # job_base_model_dir = os.path.join(BASE_MODEL_PATH, "models_{}".format(job_id), "tmp_aggregate_pars")
        job_base_model_dir = os.path.join(ResClientConfig.BASE_MODEL_PATH, "models_{}".format(job_id), "tmp_aggregate_pars")
        # 基于机器能力的动态模型融合，客户端需要接受当前中心节点的融合次数
        # latest_num = len(os.listdir(job_base_model_dir))
        latest_tmp_aggretate_file_path = os.path.join(job_base_model_dir, "avg_pars_{}".format(fed_step))
        with open(latest_tmp_aggretate_file_path, "wb") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            for line in tmp_aggregate_file.readlines():
                f.write(line)
        logger.info("recv success")
    return "ok", 200


@return_data_decorator
@app.route("/notify_dispatch/<is_dispatch>", methods=['POST'])
def notify_dispatch(is_dispatch):
    """
    is_dispatch表示模型是否下发，模型不下发时，客户端不需要加载模型接着完成下一次模型训练
    """
    if is_dispatch == "dispatch":
        # 客户端需要加载模型
        DynamicClientConfig.set_need_load_model(True)
        DynamicClientConfig.set_server_is_dispatch(True)
    else:
        DynamicClientConfig.set_need_load_model(False)
        DynamicClientConfig.set_server_is_dispatch(False)
    return "notify client success", 200


def start_communicate_client(client_ip, client_port):
    app.url_map.strict_slashes = False
    run_simple(hostname=client_ip, port=int(client_port), application=app, threaded=True)
    logger.info("galaxy learning client started")
