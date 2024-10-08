# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 18:06
# @Project : ChatTTSPlus
# @FileName: chattts_plus_pipeline.py
import os.path

import torch
import vocos
import numpy as np
import pybase16384 as b14
from numpy import dtype

from ..commons import text_utils, logger
from .. import models
from .. import trt_models
from ..commons import constants
from ..commons import norm
from ..commons.utils import RefineTextParams, InferCodeParams


class ChatTTSPlusPipeline:
    """
    ChatTTS Plus Pipeline
    """

    def __init__(self, cfg, **kwargs):
        self.logger = logger.get_logger(self.__class__.__name__)

        self.cfg = cfg
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", None)
        if self.dtype is None:
            if str(self.device) == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            # CPU 和 MPS 不支持使用 float16
            if str(self.device) != "cuda" and self.dtype == torch.float16:
                self.logger.warning("CPU and MPS do not support FLOAT16 dtype for ChatttsPlus pipeline")
                self.dtype = torch.float32
        self.load_models(**kwargs)

    def load_models(self, **kwargs):
        self.models_dict = dict()
        coef = kwargs.get("coef", None)
        if coef is None:
            coef_ = torch.rand(100)
            coef = b14.encode_to_string(coef_.numpy().astype(np.float32).tobytes())
        self.logger.info("DVAE coef: {}".format(coef))
        if "dvae_encode" in self.cfg.MODELS:
            self.cfg.MODELS["dvae_encode"]["kwargs"]["coef"] = coef
        if "dvae_decode" in self.cfg.MODELS:
            self.cfg.MODELS["dvae_decode"]["kwargs"]["coef"] = coef
        for model_name in self.cfg.MODELS:
            self.logger.info("loading model: {} >>>>".format(model_name))
            self.logger.info(self.cfg.MODELS[model_name])
            model_path_org = self.cfg.MODELS[model_name]["kwargs"]["model_path"]
            model_path_new = os.path.join(constants.CHECKPOINT_DIR, model_path_org.replace("checkpoints/", ""))
            self.cfg.MODELS[model_name]["kwargs"]["model_path"] = model_path_new
            if model_name.lower() == "vocos":
                import vocos.feature_extractors
                import vocos.models
                import vocos.heads
                feature_extractor = vocos.feature_extractors.MelSpectrogramFeatures(
                    **self.cfg.MODELS[model_name]["kwargs"]["feature_extractor_config"])
                backbone = vocos.models.VocosBackbone(**self.cfg.MODELS[model_name]["kwargs"]["backbone_config"])
                head = vocos.heads.ISTFTHead(**self.cfg.MODELS[model_name]["kwargs"]["head_config"])
                if "mps" in str(self.device):
                    device = torch.device("cpu")
                    dtype = torch.float32
                else:
                    device = self.device
                    dtype = self.dtype
                model_ = vocos.Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)
                model_.eval().to(device, dtype=dtype)
                self.models_dict[model_name] = model_
            else:
                if self.cfg.MODELS[model_name]["infer_type"] == "pytorch":
                    model_ = getattr(models, self.cfg.MODELS[model_name]["name"])(
                        **self.cfg.MODELS[model_name]["kwargs"])
                    if model_name == "tokenizer":
                        self.models_dict[model_name] = model_
                    else:
                        model_.eval().to(self.device, dtype=self.dtype)
                        self.models_dict[model_name] = model_
                elif self.cfg.MODELS[model_name]["infer_type"] == "trt":
                    raise NotImplementedError

        spk_stat_path = os.path.join(constants.CHECKPOINT_DIR, "asset/spk_stat.pt")
        self.logger.info(f"loading speaker stat: {spk_stat_path}")
        assert os.path.exists(spk_stat_path), f"Missing spk_stat.pt: {spk_stat_path}"
        spk_stat: torch.Tensor = torch.load(
            spk_stat_path,
            weights_only=True,
            mmap=True
        ).to(self.device, dtype=self.dtype)
        self.std, self.mean = spk_stat.chunk(2)

        normalizer_json = os.path.join(constants.PROJECT_DIR, "assets", "homophones_map.json")
        self.logger.info(f"loading normalizer: {normalizer_json}")
        self.normalizer = norm.Normalizer(normalizer_json)

    @torch.no_grad()
    def infer(self,
              text,
              stream=False,
              lang=None,
              skip_refine_text=False,
              refine_text_only=False,
              use_decoder=True,
              do_text_normalization=True,
              do_homophone_replacement=True,
              params_refine_text=RefineTextParams(),
              params_infer_code=InferCodeParams(),
              **kwargs):
        pass
