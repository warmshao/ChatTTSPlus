# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 18:06
# @Project : ChatTTSPlus
# @FileName: chattts_plus_pipeline.py

from ..commons import text_utils, logger


class ChatTTSPlusPipeline:
    """
    ChatTTS Plus Pipeline
    """

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.init_pipeline(**kwargs)

    def init_pipeline(self, **kwargs):
        self.logger = logger.get_logger(self.__class__.__name__)
        self.load_models(**kwargs)

    def load_models(self, **kwargs):
        self.model_dict = dict()
        for model_name in self.cfg.MODELS:
            pass

