# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 18:06
# @Project : ChatTTSPlus
# @FileName: chattts_plus_pipeline.py
import torch

from ..commons import text_utils, logger
from .. import models
from .. import trt_models


class ChatTTSPlusPipeline:
    """
    ChatTTS Plus Pipeline
    """

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.device = kwargs.get("device", "cpu")
        if self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.logger = logger.get_logger(self.__class__.__name__)
        self.load_models(**kwargs)

    def load_models(self, **kwargs):
        self.model_dict = dict()
        coef = kwargs.get("coef", torch.rand(100))
        if "dvae_encode" in self.cfg.MODELS:
            self.cfg.MODELS["dvae_encode"]["kwargs"]["coef"] = coef
        if "dvae_decode" in self.cfg.MODELS:
            self.cfg.MODELS["dvae_decode"]["kwargs"]["coef"] = coef
        for model_name in self.cfg.MODELS:
            self.logger.info("loading model: {} >>>>".format(model_name))
            self.logger.info(self.cfg.MODELS[model_name])
            if model_name.lower() == "vocos":
                pass
            else:
                if self.cfg.MODELS[model_name]["infer_type"] == "pytorch":
                    model_ = getattr(models, self.cfg.MODELS[model_name]["name"])(
                        **self.cfg.MODELS[model_name]["kwargs"])
                    model_.eval()
                    self.model_dict[model_name] = model_
                elif self.cfg.MODELS[model_name]["infer_type"] == "trt":
                    pass

    def infer(self):
        pass
