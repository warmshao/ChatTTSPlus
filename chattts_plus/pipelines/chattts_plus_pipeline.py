# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 18:06
# @Project : ChatTTSPlus
# @FileName: chattts_plus_pipeline.py
import os.path
import pdb
import time
import lzma
import torch
import torchaudio
import vocos
import numpy as np
import pybase16384 as b14
from numpy import dtype
from typing import Literal, Optional, List, Tuple, Dict, Union

from ..commons import text_utils, logger
from .. import models
from .. import trt_models
from ..commons import constants
from ..commons import norm
from ..commons.utils import RefineTextParams, InferCodeParams
from ..models import processors


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
        self.logger.info(f"device: {str(self.device)}")
        self.logger.info(f"dtype: {str(self.dtype)}")
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
                model_.load_state_dict(
                    torch.load(self.cfg.MODELS[model_name]["kwargs"]["model_path"], weights_only=True, mmap=True))
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
                    model_ = getattr(trt_models, self.cfg.MODELS[model_name]["name"])(
                        **self.cfg.MODELS[model_name]["kwargs"])
                    model_.eval().to(self.device, dtype=self.dtype)
                    self.models_dict[model_name] = model_

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
    def _infer_code(
            self,
            text,
            stream: bool,
            return_hidden: bool,
            params: InferCodeParams,
    ):
        self.logger.info("Start inference audio code >>>>")
        if not isinstance(text, list):
            text = [text]

        assert len(text), "text should not be empty"

        if not isinstance(params.temperature, list):
            temperature = [params.temperature] * self.models_dict["gpt"].num_vq
        else:
            temperature = params.temperature

        for i, t in enumerate(text):
            text[i] = (
                t.replace("[Stts]", "")
                .replace("[spk_emb]", "")
                .replace("[empty_spk]", "")
                .strip()
            )
            """
            see https://github.com/2noise/ChatTTS/issues/459
            """

        if params.prompt:
            text = [params.prompt + i for i in text]

        txt_smp = "" if params.txt_smp is None else params.txt_smp
        if params.spk_emb is not None:
            text = [f"[Stts][spk_emb]{txt_smp}{i}[Ptts]" for i in text]
        else:
            text = [f"[Stts][empty_spk]{txt_smp}{i}[Ptts]" for i in text]
        input_ids, attention_mask, text_mask = self.models_dict["tokenizer"].encode(
            text,
            self.models_dict["gpt"].num_vq,
            prompt_str=params.spk_smp,
            device=self.device
        )

        emb = self.models_dict["gpt"](input_ids, text_mask)

        if params.spk_emb is not None:
            self.models_dict["tokenizer"].apply_spk_emb(
                emb, params.spk_emb, input_ids, self.device
            )

        num_code = int(self.models_dict["gpt"].emb_code[0].num_embeddings - 1)

        logits_warpers, logits_processors = processors.gen_logits(
            num_code=num_code,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )

        result = self.models_dict["gpt"].generate(
            emb,
            input_ids,
            temperature=torch.tensor(temperature, device=self.device),
            eos_token=num_code,
            attention_mask=attention_mask,
            max_new_token=params.max_new_token,
            min_new_token=params.min_new_token,
            logits_warpers=logits_warpers,
            logits_processors=logits_processors,
            infer_text=False,
            return_hidden=return_hidden,
            stream=stream,
            show_tqdm=params.show_tqdm,
            ensure_non_empty=params.ensure_non_empty,
            stream_batch=params.stream_batch,
        )
        return result

    @torch.no_grad()
    def _refine_text(
            self,
            text: str,
            params: RefineTextParams,
    ):

        text = [f"[Sbreak]{i}[Pbreak]{params.prompt}" for i in text]

        input_ids, attention_mask, text_mask = self.models_dict["tokenizer"].encode(
            text,
            self.models_dict["gpt"].num_vq,
            device=self.device,
        )

        logits_warpers, logits_processors = processors.gen_logits(
            num_code=self.models_dict["tokenizer"].len,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )
        emb = self.models_dict["gpt"](input_ids, text_mask)

        result = next(
            self.models_dict["gpt"].generate(
                emb,
                input_ids,
                temperature=torch.tensor([params.temperature], device=self.device),
                eos_token=self.models_dict["tokenizer"].eos_token,
                attention_mask=attention_mask,
                max_new_token=params.max_new_token,
                min_new_token=params.min_new_token,
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
                infer_text=True,
                stream=False,
                show_tqdm=params.show_tqdm,
                ensure_non_empty=params.ensure_non_empty
            )
        )
        return result

    @torch.inference_mode()
    def sample_audio_speaker(self, wav: Union[np.ndarray, torch.Tensor]) -> str:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).to(self.device, dtype=self.dtype)
        return self.models_dict["tokenizer"]._encode_prompt(self.models_dict["dvae_encode"](wav, "encode").squeeze_(0))

    @torch.inference_mode()
    def _decode_to_wavs(
            self,
            result_list,
            use_decoder: bool,
    ):
        self.logger.info("Start decode to wavs >>>>")
        decoder = self.models_dict["dvae_decode"] if use_decoder else self.models_dict["dvae_encode"]
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
        mel_specs = decoder(batch_result).to(dtype=next(self.models_dict["vocos"].parameters()).dtype)
        if "mps" in str(mel_specs.device):
            mel_specs = mel_specs.to(device=torch.device("cpu"))
        wavs = self.models_dict["vocos"].decode(mel_specs)
        return wavs

    def sample_random_speaker(self) -> str:
        return self._encode_spk_emb(self._sample_random_speaker())

    @staticmethod
    @torch.no_grad()
    def _encode_spk_emb(spk_emb: torch.Tensor) -> str:
        arr: np.ndarray = spk_emb.to(dtype=torch.float16, device="cpu").numpy()
        s = b14.encode_to_string(
            lzma.compress(
                arr.tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
        )
        return s

    @torch.no_grad()
    def _sample_random_speaker(self) -> torch.Tensor:
        dim: int = self.std.shape[-1]
        spk = (
            torch.randn(dim, device=self.std.device, dtype=self.std.dtype)
            .mul_(self.std)
            .add_(self.mean)
        )
        return spk

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
            self.logger.info("Optimization on text, such as split, merge and so on")
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

            for ti in range(len(retext)):
                if not retext[ti].strip().endswith("[uv_break]"):
                    retext[ti] += " [uv_break]"
            text = retext
            self.logger.info("Finish text optimization: ")
            self.logger.info(text)

        text = [
            self.normalizer(
                t,
                do_text_normalization,
                do_homophone_replacement,
                lang,
            )
            for t in text
        ]
        self.logger.info("Finish text normalization: ")
        self.logger.info(text)

        # refine text
        if not skip_refine_text:
            self.logger.info("Process Text Refinement >>>")
            refined = self._refine_text(
                text,
                params_refine_text
            )
            text_tokens = refined.ids
            text_tokens = [i[i.less(self.models_dict["tokenizer"].break_0_ids)] for i in text_tokens]
            text = self.models_dict["tokenizer"].decode(text_tokens)
            self.logger.info("Refine text: ")
            self.logger.info(text)
            if refine_text_only:
                yield text
                return

        if stream:
            length = 0
            pass_batch_count = 0
        for result in self._infer_code(
                text,
                stream,
                use_decoder,
                params_infer_code,
        ):
            wavs = self._decode_to_wavs(
                result.hiddens if use_decoder else result.ids,
                use_decoder,
            )
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
        if kwargs.get("speaker_audio_path", None) is not None:
            speaker_audio_path = kwargs.get("speaker_audio_path", None)
            assert os.path.exists(speaker_audio_path), f"speaker_audio_path {speaker_audio_path} not exists!"
            speaker_audio_text = kwargs.get("speaker_audio_text", "")
            self.logger.info("Use zero shot >>>")
            self.logger.info(f"speaker_audio_path is {speaker_audio_path}")
            self.logger.info(f"speaker_audio_text is {speaker_audio_text}")
            audio_wav, audio_sr_ = torchaudio.load(speaker_audio_path)
            audio_sr = 24000
            audio_wav = torchaudio.functional.resample(audio_wav, orig_freq=audio_sr_, new_freq=audio_sr)
            audio_wav = torch.mean(audio_wav, 0).to(self.device, dtype=self.dtype)
            spk_smp = self.sample_audio_speaker(audio_wav)
            params_infer_code.txt_smp = speaker_audio_text
            params_infer_code.spk_smp = spk_smp
            params_infer_code.spk_emb = None
        elif kwargs.get("speaker_emb_path", None) is not None:
            speaker_emb_path = kwargs.get("speaker_emb_path", None)
            assert os.path.exists(speaker_emb_path), f"speaker_emb_path {speaker_emb_path} not exists!"
            self.logger.info(f"loading speaker_emb from {speaker_emb_path}")
            speaker_emb = torch.load(speaker_emb_path)
            params_infer_code.spk_emb = speaker_emb
        else:
            self.logger.info("speaker_emb is None, random select a speaker!")
            speaker_emb = self.sample_random_speaker()
            params_infer_code.spk_emb = speaker_emb
            self.logger.info(f"speaker embedding is : {speaker_emb}")
            SPEAKER_DIR = kwargs.get("speaker_save_dir",
                                     os.path.join(os.path.dirname(__file__), "..", "..", "results/speakers"))
            os.makedirs(SPEAKER_DIR, exist_ok=True)
            torch.save(speaker_emb, f"{SPEAKER_DIR}/{time.time()}.pt")
            self.logger.info(f"saving speaker emb at: {SPEAKER_DIR}/{time.time()}.pt")

        self.logger.info("Params refine text:")
        self.logger.info(params_refine_text.__dict__)
        self.logger.info("Params infer code:")
        self.logger.info(params_infer_code.__dict__)

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
