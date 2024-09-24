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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
import json

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import os

import cv2
import yaml
import paddle.distributed as dist
import numpy as np
from ppocr.utils.utility import print_dict
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import WandbLogger, Loggers
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

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
        self.add_argument("-c", "--config", help="configuration file to use",default='./recconfig.yaml')
        self.add_argument("-o", "--opt", nargs="+", help="set configuration options")
        self.add_argument(
            "-p",
            "--profiler_options",
            type=str,
            default=None,
            help="The option of profiler, which should be in format "
            '"key1=value1;key2=value2;key3=value3".',
        )

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
class rec:
    def __init__(self):
        self.configpath = './recconfig.yaml'
        self.config, self.device, self.logger, self.vdl_writer = self.preprocess()
        self.rec_init()

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


    def rec_init(self):
        global_config = self.config["Global"]

        # build post process
        self.post_process_class = build_post_process(self.config["PostProcess"], global_config)

        # build model
        if hasattr(self.post_process_class, "character"):
            char_num = len(getattr(self.post_process_class, "character"))
            if self.config["Architecture"]["algorithm"] in [
                "Distillation",
            ]:  # distillation model
                for key in self.config["Architecture"]["Models"]:
                    if (
                        self.config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                    ):  # multi head
                        out_channels_list = {}
                        if self.config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                            char_num = char_num - 2
                        if self.config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                            char_num = char_num - 3
                        out_channels_list["CTCLabelDecode"] = char_num
                        out_channels_list["SARLabelDecode"] = char_num + 2
                        out_channels_list["NRTRLabelDecode"] = char_num + 3
                        self.config["Architecture"]["Models"][key]["Head"][
                            "out_channels_list"
                        ] = out_channels_list
                    else:
                        self.config["Architecture"]["Models"][key]["Head"][
                            "out_channels"
                        ] = char_num
            elif self.config["Architecture"]["Head"]["name"] == "MultiHead":  # multi head
                out_channels_list = {}
                char_num = len(getattr(self.post_process_class, "character"))
                if self.config["PostProcess"]["name"] == "SARLabelDecode":
                    char_num = char_num - 2
                if self.config["PostProcess"]["name"] == "NRTRLabelDecode":
                    char_num = char_num - 3
                out_channels_list["CTCLabelDecode"] = char_num
                out_channels_list["SARLabelDecode"] = char_num + 2
                out_channels_list["NRTRLabelDecode"] = char_num + 3
                self.config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
            else:  # base rec model
                self.config["Architecture"]["Head"]["out_channels"] = char_num

        if self.config["Architecture"].get("algorithm") in ["LaTeXOCR"]:
            self.config["Architecture"]["Backbone"]["is_predict"] = True
            self.config["Architecture"]["Backbone"]["is_export"] = True
            self.config["Architecture"]["Head"]["is_export"] = True

        self.model = build_model(self.config["Architecture"])

        load_model(self.config, self.model)

        # create data ops
        transforms = []
        for op in self.config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            elif op_name in ["RecResizeImg"]:
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                if self.config["Architecture"]["algorithm"] == "SRN":
                    op[op_name]["keep_keys"] = [
                        "image",
                        "encoder_word_pos",
                        "gsrm_word_pos",
                        "gsrm_slf_attn_bias1",
                        "gsrm_slf_attn_bias2",
                    ]
                elif self.config["Architecture"]["algorithm"] == "SAR":
                    op[op_name]["keep_keys"] = ["image", "valid_ratio"]
                elif self.config["Architecture"]["algorithm"] == "RobustScanner":
                    op[op_name]["keep_keys"] = ["image", "valid_ratio", "word_positons"]
                else:
                    op[op_name]["keep_keys"] = ["image"]
            transforms.append(op)
        global_config["infer_mode"] = True
        self.ops = create_operators(transforms, global_config)

        save_res_path = self.config["Global"].get(
            "save_res_path", "./output/rec/predicts_rec.txt"
        )
        if not os.path.exists(os.path.dirname(save_res_path)):
            os.makedirs(os.path.dirname(save_res_path))

        self.model.eval()
    def rec_infer(self,img):
        data = {"image": img}
        batch = transform(data, self.ops)
        if self.config["Architecture"]["algorithm"] == "SRN":
            encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
            gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
            gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
            gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

            others = [
                paddle.to_tensor(encoder_word_pos_list),
                paddle.to_tensor(gsrm_word_pos_list),
                paddle.to_tensor(gsrm_slf_attn_bias1_list),
                paddle.to_tensor(gsrm_slf_attn_bias2_list),
            ]
        if self.config["Architecture"]["algorithm"] == "SAR":
            valid_ratio = np.expand_dims(batch[-1], axis=0)
            img_metas = [paddle.to_tensor(valid_ratio)]
        if self.config["Architecture"]["algorithm"] == "RobustScanner":
            valid_ratio = np.expand_dims(batch[1], axis=0)
            word_positons = np.expand_dims(batch[2], axis=0)
            img_metas = [
                paddle.to_tensor(valid_ratio),
                paddle.to_tensor(word_positons),
            ]
        if self.config["Architecture"]["algorithm"] == "CAN":
            image_mask = paddle.ones(
                (np.expand_dims(batch[0], axis=0).shape), dtype="float32"
            )
            label = paddle.ones((1, 36), dtype="int64")
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        if self.config["Architecture"]["algorithm"] == "SRN":
            preds = self.model(images, others)
        elif self.config["Architecture"]["algorithm"] == "SAR":
            preds = self.model(images, img_metas)
        elif self.config["Architecture"]["algorithm"] == "RobustScanner":
            preds = self.model(images, img_metas)
        elif self.config["Architecture"]["algorithm"] == "CAN":
            preds = self.model([images, image_mask, label])
        else:
            preds = self.model(images)
        post_result = self.post_process_class(preds)
        info = None
        if isinstance(post_result, dict):
            rec_info = dict()
            for key in post_result:
                if len(post_result[key][0]) >= 2:
                    rec_info[key] = {
                        "label": post_result[key][0][0],
                        "score": float(post_result[key][0][1]),
                    }
            info = json.dumps(rec_info, ensure_ascii=False)
        elif isinstance(post_result, list) and isinstance(post_result[0], int):
            # for RFLearning CNT branch
            info = str(post_result[0])
        elif self.config["Architecture"]["algorithm"] == "LaTeXOCR":
            info = str(post_result[0])
        else:
            if len(post_result[0]) >= 2:
                info = post_result[0][0] + "\t" + str(post_result[0][1])

        if info is not None:
            self.logger.info("\t result: {}".format(info))
        if float(info.split('\t')[-1])<=0.5:
            return "无有效内容"
        return info.split('\t')[0]


if __name__ == "__main__":
    filelist = ['../test1.jpg_polygon_12_0.png','../test1.jpg_polygon_12_0.png','../test1.jpg_polygon_12_0.png','../test1.jpg_polygon_12_0.png','../test1.jpg_polygon_12_0.png']
    detinstence = rec()
    detinstence.rec_init()
    for i in filelist:
        img = cv2.imread(i)
        success, encoded_image = cv2.imencode('.png', img)
        print(detinstence.rec_infer(encoded_image.tobytes()))
