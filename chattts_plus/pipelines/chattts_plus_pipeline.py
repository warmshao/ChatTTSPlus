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
    def _refine_text(
            self,
            text: str,
            params: RefineTextParams,
    ):

        gpt = self.models_dict["gpt"]

        text = [f"[Sbreak]{i}[Pbreak]{params.prompt}" for i in text]

        input_ids, attention_mask, text_mask = self.tokenizer.encode(
            text,
            self.gpt.num_vq,
            device=gpt.device_gpt,
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.tokenizer.len,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )

        emb = gpt(input_ids, text_mask)

        result = next(
            gpt.generate(
                emb,
                input_ids,
                temperature=torch.tensor([params.temperature], device=device),
                eos_token=self.tokenizer.eos_token,
                attention_mask=attention_mask,
                max_new_token=params.max_new_token,
                min_new_token=params.min_new_token,
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
                infer_text=True,
                stream=False,
                show_tqdm=params.show_tqdm,
                ensure_non_empty=params.ensure_non_empty,
                context=self.context,
            )
        )
        return result

    def _infer(
            self,
            text,
            stream=False,
            lang=None,
            skip_refine_text=False,
            refine_text_only=False,
            use_decoder=True,
            do_text_normalization=True,
            do_text_optimization=True,
            do_homophone_replacement=True,
            params_refine_text=RefineTextParams(),
            params_infer_code=InferCodeParams(),
            **kwargs
    ):

        if not isinstance(text, list):
            text = [text]

        # 参考chattts-ui做分割、合并以及数字转换等优化
        if do_text_optimization:
            text_list = []
            for text_ in text:
                text_list.extend([t.strip() for t in text_.split("\n") if t.strip()])
            new_text = text_utils.split_text(text_list)
            retext = []
            short_text = ""
            for it in new_text:
                if len(it) < 30:
                    short_text += f"{it} [uv_break] "
                    if len(short_text) > 30:
                        retext.append(short_text)
                        short_text = ""
                else:
                    retext.append(short_text + it)
                    short_text = ""
            if len(short_text) > 30 or len(retext) < 1:
                retext.append(short_text)
            elif short_text:
                retext[-1] += f" [uv_break] {short_text}"

            for text_ in retext:
                if not text_.strip().endswith("[uv_break]"):
                    text_ += " [uv_break]"
            text = retext

        text = [
            self.normalizer(
                t,
                do_text_normalization,
                do_homophone_replacement,
                lang,
            )
            for t in text
        ]

        # refine text
        if not skip_refine_text:
            refined = self._refine_text(
                text,
                self.device,
                params_refine_text,
            )
            text_tokens = refined.ids
            text_tokens = [i[i.less(self.tokenizer.break_0_ids)] for i in text_tokens]
            text = self.tokenizer.decode(text_tokens)
            refined.destroy()
            if refine_text_only:
                yield text
                return

        if stream:
            length = 0
            pass_batch_count = 0
        for result in self._infer_code(
                text,
                stream,
                self.device,
                use_decoder,
                params_infer_code,
        ):
            wavs = self._decode_to_wavs(
                result.hiddens if use_decoder else result.ids,
                use_decoder,
            )
            result.destroy()
            if stream:
                pass_batch_count += 1
                if pass_batch_count <= params_infer_code.pass_first_n_batches:
                    continue
                a = length
                b = a + params_infer_code.stream_speed
                if b > wavs.shape[1]:
                    b = wavs.shape[1]
                new_wavs = wavs[:, a:b]
                length = b
                yield new_wavs
            else:
                yield wavs
        if stream:
            new_wavs = wavs[:, length:]
            # Identify rows with non-zero elements using np.any
            # keep_rows = np.any(array != 0, axis=1)
            keep_cols = np.sum(new_wavs != 0, axis=0) > 0
            # Filter both rows and columns using slicing
            yield new_wavs[:][:, keep_cols]

    @torch.no_grad()
    def infer(self,
              text,
              stream=False,
              lang=None,
              skip_refine_text=False,
              refine_text_only=False,
              use_decoder=True,
              do_text_normalization=True,
              do_text_optimization=True,
              do_homophone_replacement=True,
              params_refine_text=RefineTextParams(),
              params_infer_code=InferCodeParams(),
              **kwargs):
        res_gen = self._infer(
            text,
            stream,
            lang,
            skip_refine_text,
            refine_text_only,
            use_decoder,
            do_text_normalization,
            do_text_optimization,
            do_homophone_replacement,
            params_refine_text,
            params_infer_code,
            **kwargs
        )
        if stream:
            return res_gen
        else:
            return next(res_gen)
