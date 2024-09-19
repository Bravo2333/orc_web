# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import json

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

import os
import yaml
import paddle
import paddle.distributed as dist
import cv2
import numpy as np
from ppocr.utils.utility import print_dict, AverageMeter
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import WandbLogger, Loggers

def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config
class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use",default='./detconfig.yaml')
        self.add_argument("-o", "--opt", nargs="+", help="set configuration options")
        self.add_argument(
            "-p",
            "--profiler_options",
            type=str,
            default=None,
            help="The option of profiler, which should be in format "
            '"key1=value1;key2=value2;key3=value3".',
        )

    def draw_det_res(dt_boxes, config, img, img_name, save_path):
        import cv2

        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        # logger.info("The detected Image saved in {}".format(save_path))
    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config
def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))
def draw_det_res(dt_boxes, config, img, img_name, save_path):
    import cv2

    src_im = img
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(img_name))
    cv2.imwrite(save_path, src_im)
    # logger.info("The detected Image saved in {}".format(save_path))
def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split(".")
            assert sub_keys[0] in config, (
                "the sub_keys can only be one of global_config: {}, but get: "
                "{}, please check your running command".format(
                    config.keys(), sub_keys[0]
                )
            )
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config

class det:
    def __init__(self):
        self.configpath = './detconfig.yaml'
        self.config, self.device, self.logger, self.vdl_writer = self.preprocess()
    def preprocess(self,is_train=False):
        # FLAGS = ArgsParser().parse_args()
        # profiler_options = FLAGS.profiler_options
        config = load_config(self.configpath)
        # config = merge_config(config, FLAGS.opt)
        # config = merge_config(config, profile_dic)

        # if is_train:
        #     # save_config
        #     save_model_dir = config["Global"]["save_model_dir"]
        #     os.makedirs(save_model_dir, exist_ok=True)
        #     with open(os.path.join(save_model_dir, "config.yml"), "w") as f:
        #         yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)
        #     log_file = "{}/train.log".format(save_model_dir)
        # else:
        #     log_file = None
        log_file = None
        log_ranks = config["Global"].get("log_ranks", "0")
        logger = get_logger(log_file=log_file, log_ranks=log_ranks)

        # check if set use_gpu=True in paddlepaddle cpu version
        # use_gpu = config["Global"].get("use_gpu", False)
        # use_xpu = config["Global"].get("use_xpu", False)
        # use_npu = config["Global"].get("use_npu", False)
        # use_mlu = config["Global"].get("use_mlu", False)
        use_gpu = True

        alg = config["Architecture"]["algorithm"]
        assert alg in [
            "EAST",
            "DB",
            "SAST",
            "Rosetta",
            "CRNN",
            "STARNet",
            "RARE",
            "SRN",
            "CLS",
            "PGNet",
            "Distillation",
            "NRTR",
            "TableAttn",
            "SAR",
            "PSE",
            "SEED",
            "SDMGR",
            "LayoutXLM",
            "LayoutLM",
            "LayoutLMv2",
            "PREN",
            "FCE",
            "SVTR",
            "SVTR_LCNet",
            "ViTSTR",
            "ABINet",
            "DB++",
            "TableMaster",
            "SPIN",
            "VisionLAN",
            "Gestalt",
            "SLANet",
            "RobustScanner",
            "CT",
            "RFL",
            "DRRG",
            "CAN",
            "Telescope",
            "SATRN",
            "SVTR_HGNet",
            "ParseQ",
            "CPPD",
            "LaTeXOCR",
        ]

        device = "gpu:{}".format(dist.ParallelEnv().dev_id) if use_gpu else "cpu"

        device = paddle.set_device(device)

        config["Global"]["distributed"] = dist.get_world_size() != 1

        loggers = []

        # if "use_visualdl" in config["Global"] and config["Global"]["use_visualdl"]:
        #     logger.warning(
        #         "You are using VisualDL, the VisualDL is deprecated and "
        #         "removed in ppocr!"
        #     )
        #     log_writer = None
        # if (
        #     "use_wandb" in config["Global"] and config["Global"]["use_wandb"]
        # ) or "wandb" in config:
        #     save_dir = config["Global"]["save_model_dir"]
        #     wandb_writer_path = "{}/wandb".format(save_dir)
        #     if "wandb" in config:
        #         wandb_params = config["wandb"]
        #     else:
        #         wandb_params = dict()
        #     wandb_params.update({"save_dir": save_dir})
        #     log_writer = WandbLogger(**wandb_params, config=config)
        #     loggers.append(log_writer)
        # else:
        #     log_writer = None
        print_dict(config, logger)

        if loggers:
            log_writer = Loggers(loggers)
        else:
            log_writer = None

        logger.info("train with paddle {} and device {}".format(paddle.__version__, device))
        return config, device, logger, log_writer




    @paddle.no_grad()
    def det(self,config,logger,filelist):
        global_config = config["Global"]

        # build model
        model = build_model(config["Architecture"])

        load_model(config, model)
        # build post process
        post_process_class = build_post_process(config["PostProcess"])

        # create data ops
        transforms = []
        for op in config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image", "shape"]
            transforms.append(op)

        ops = create_operators(transforms, global_config)

        save_res_path = config["Global"]["save_res_path"]
        if not os.path.exists(os.path.dirname(save_res_path)):
            os.makedirs(os.path.dirname(save_res_path))

        model.eval()
        with open(save_res_path, "wb") as fout:
            for file in filelist:
                logger.info("infer_img: {}".format(file))
                with open(file, "rb") as f:
                    img = f.read()
                    data = {"image": img}
                batch = transform(data, ops)

                images = np.expand_dims(batch[0], axis=0)
                shape_list = np.expand_dims(batch[1], axis=0)
                images = paddle.to_tensor(images)
                preds = model(images)# 计算
                post_result = post_process_class(preds, shape_list)

                src_img = cv2.imread(file)

                dt_boxes_json = []
                # parser boxes if post_result is dict
                if isinstance(post_result, dict):
                    det_box_json = {}
                    for k in post_result.keys():
                        boxes = post_result[k][0]["points"]
                        dt_boxes_list = []
                        for box in boxes:
                            tmp_json = {"transcription": ""}
                            tmp_json["points"] = np.array(box).tolist()
                            dt_boxes_list.append(tmp_json)
                        det_box_json[k] = dt_boxes_list
                        save_det_path = os.path.dirname(
                            config["Global"]["save_res_path"]
                        ) + "/det_results_{}/".format(k)
                        draw_det_res(boxes, config, src_img, file, save_det_path)
                else:
                    boxes = post_result[0]["points"]
                    dt_boxes_json = []
                    # write result
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json["points"] = np.array(box).tolist()
                        dt_boxes_json.append(tmp_json)
                    save_det_path = (
                        os.path.dirname(config["Global"]["save_res_path"]) + "/det_results/"
                    )
                    draw_det_res(boxes, config, src_img, file, save_det_path)

                otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
                fout.write(otstr.encode())

        logger.info("success!")


if __name__ == "__main__":
    filelist = ['../test1.jpg_polygon_12_0.png']
    detinstence = det()
    detinstence.det(detinstence.config,detinstence.logger,filelist)