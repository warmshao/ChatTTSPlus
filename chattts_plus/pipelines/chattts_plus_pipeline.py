# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 18:06
# @Project : ChatTTSPlus
# @FileName: chattts_plus_pipeline.py
import lzma
import os.path
import pdb
import time
from typing import Union

import numpy as np
import pybase16384 as b14
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .. import models
from .. import trt_models
from ..commons import constants
from ..commons import norm
from ..commons import text_utils, logger
from ..commons.onnx2trt import convert_onnx_to_trt
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
        self.load_lora = False

    def load_models(self, **kwargs):
        self.models_dict = dict()
        coef = kwargs.get("coef", None)
        self.infer_type = None
        if coef is None:
            coef_ = torch.rand(100)
            coef = b14.encode_to_string(coef_.numpy().astype(np.float32).tobytes())
        self.dave_coef = coef
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
            if not os.path.exists(model_path_new):
                self.logger.warn(f"{model_path_new} not exists! Need to download from HuggingFace")
                hf_hub_download(repo_id="2Noise/ChatTTS", subfolder="asset",
                                filename=os.path.basename(model_path_new),
                                local_dir=constants.CHECKPOINT_DIR)
                self.logger.info(f"download {model_path_new} from 2Noise/ChatTTS")
            self.cfg.MODELS[model_name]["kwargs"]["model_path"] = model_path_new
            trt_model_path_org = self.cfg.MODELS[model_name]["kwargs"].get("trt_model_path", None)
            if trt_model_path_org is not None:
                trt_model_path_new = os.path.join(constants.CHECKPOINT_DIR,
                                                  trt_model_path_org.replace("checkpoints/", ""))
                self.cfg.MODELS[model_name]["kwargs"]["trt_model_path"] = trt_model_path_new
                if not os.path.exists(trt_model_path_new):
                    self.logger.warn(f"{trt_model_path_new} not exists! Need to download from HuggingFace")
                    onnx_model_path_new = trt_model_path_new[:-4] + ".onnx"
                    hf_hub_download(repo_id="warmshao/ChatTTSPlus",
                                    filename=os.path.basename(onnx_model_path_new),
                                    local_dir=constants.CHECKPOINT_DIR)
                    self.logger.info(f"download {onnx_model_path_new} from 2Noise/ChatTTS")
                    self.logger.info(f"Now convert {onnx_model_path_new} to trt")
                    convert_onnx_to_trt(onnx_model_path_new, trt_model_path_new)
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
                    if not self.infer_type:
                        self.infer_type = self.cfg.MODELS[model_name]["infer_type"]
                    model_ = getattr(models, self.cfg.MODELS[model_name]["name"])(
                        **self.cfg.MODELS[model_name]["kwargs"])
                    if model_name == "tokenizer":
                        self.models_dict[model_name] = model_
                    else:
                        model_.eval().to(self.device, dtype=self.dtype)
                        self.models_dict[model_name] = model_
                elif self.cfg.MODELS[model_name]["infer_type"] == "trt":
                    if not self.infer_type:
                        self.infer_type = self.cfg.MODELS[model_name]["infer_type"]
                    model_ = getattr(trt_models, self.cfg.MODELS[model_name]["name"])(
                        **self.cfg.MODELS[model_name]["kwargs"])
                    model_.eval().to(self.device, dtype=self.dtype)
                    self.models_dict[model_name] = model_

        spk_stat_path = os.path.join(constants.CHECKPOINT_DIR, "asset/spk_stat.pt")
        if not os.path.exists(spk_stat_path):
            self.logger.warning(f"{spk_stat_path} not exists! Need to download from HuggingFace")
            hf_hub_download(repo_id="2Noise/ChatTTS", subfolder="asset",
                            filename=os.path.basename(spk_stat_path),
                            local_dir=constants.CHECKPOINT_DIR)
            self.logger.info(f"download {spk_stat_path} from 2Noise/ChatTTS")
        self.logger.info(f"loading speaker stat: {spk_stat_path}")
        assert os.path.exists(spk_stat_path), f"Missing spk_stat.pt: {spk_stat_path}"
        spk_stat: torch.Tensor = torch.load(
            spk_stat_path,
            weights_only=True,
            mmap=True
        ).to(self.device, dtype=self.dtype)
        self.std, self.mean = spk_stat.chunk(2)

        normalizer_json = os.path.join(constants.CHECKPOINT_DIR, "homophones_map.json")
        if not os.path.exists(normalizer_json):
            self.logger.warning(f"{normalizer_json} not exists! Need to download from HuggingFace")
            hf_hub_download(repo_id="warmshao/ChatTTSPlus",
                            filename=os.path.basename(normalizer_json),
                            local_dir=constants.CHECKPOINT_DIR)
            self.logger.info(f"download {normalizer_json} from warmshao/ChatTTSPlus")
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
        return self.models_dict["tokenizer"]._encode_prompt(
            self.models_dict["dvae_encode"](wav[None], "encode").squeeze_(0))

    @torch.inference_mode()
    def _decode_to_wavs(
            self,
            result_list,
            use_decoder: bool,
    ):
        self.logger.info("Start decode to wavs >>>>")
        decoder = self.models_dict["dvae_decode"] if use_decoder else self.models_dict["dvae_encode"]
        wavs = []
        if len(result_list) == 0:
            return wavs

        for i in range(len(result_list)):
            src = result_list[i].permute(1, 0)
            mel_specs = decoder(src[None]).to(dtype=next(self.models_dict["vocos"].parameters()).dtype)
            if "mps" in str(mel_specs.device):
                mel_specs = mel_specs.to(device=torch.device("cpu"))
            wav_ = self.models_dict["vocos"].decode(mel_specs)
            wavs.append(wav_[0])
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
            text_in,
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

        if not isinstance(text_in, list):
            text_in = [text_in]

        # 参考chattts-ui做分割、合并以及数字转换等优化
        if do_text_optimization:
            self.logger.info("Optimization on text, such as split, merge and so on")
            text_list = []
            for text_ in text_in:
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

            text_in = retext
            self.logger.info("Finish text optimization: ")
            self.logger.info(text_in)

        text_in = [
            self.normalizer(
                t,
                do_text_normalization,
                do_homophone_replacement,
                lang,
            )
            for t in text_in
        ]
        self.logger.info("Finish text normalization: ")
        self.logger.info(text_in)

        slice_size = kwargs.get("slice_size", 4)
        if len(text_in) > slice_size:
            self.logger.warning(
                f"len of text is {len(text_in)} > 4, only support max batch size is equal to 4, so we need to slice to inference")

        for ii in tqdm(range(0, len(text_in), slice_size)):
            text = text_in[ii:ii + slice_size].copy()
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

            if not refine_text_only:
                for ti in range(len(text)):
                    if not text[ti].strip().endswith("[uv_break]"):
                        text[ti] += " [uv_break]"
                if stream:
                    length = 0
                    pass_batch_count = 0
                if kwargs.get("lora_path", None):
                    if self.infer_type == "pytorch":
                        from peft import PeftModel
                        import copy
                        self.logger.info(f"load lora into gpt: {kwargs.get('lora_path')}")
                        self.models_dict['gpt'].gpt_org = self.models_dict['gpt'].gpt.cpu()
                        peft_model = PeftModel.from_pretrained(copy.deepcopy(self.models_dict['gpt'].gpt),
                                                               kwargs.get("lora_path"),
                                                               device_map=self.device,
                                                               torch_dtype=self.dtype)
                        peft_model.config.use_cache = True
                        peft_model = peft_model.merge_and_unload()
                        self.models_dict['gpt'].gpt = peft_model.to(self.device, dtype=self.dtype)
                    else:
                        self.logger.error("Lora only support pytorch Now!")
                for result in self._infer_code(
                        text,
                        stream,
                        use_decoder,
                        params_infer_code
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
                if kwargs.get("lora_path", None):
                    if self.infer_type == "pytorch":
                        self.logger.info("unload lora !")
                        del self.models_dict['gpt'].gpt
                        torch.cuda.empty_cache()
                        self.models_dict['gpt'].gpt = self.models_dict['gpt'].gpt_org.to(self.device, dtype=self.dtype)

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
        if kwargs.get("speaker_audio_path", None):
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
        elif kwargs.get("speaker_emb_path", None):
            speaker_emb_path = kwargs.get("speaker_emb_path", None)
            assert os.path.exists(speaker_emb_path), f"speaker_emb_path {speaker_emb_path} not exists!"
            self.logger.info(f"loading speaker_emb from {speaker_emb_path}")
            try:
                # 尝试不同的加载方式
                try:
                    # 首先尝试使用 weights_only=True
                    speaker_emb = torch.load(speaker_emb_path, weights_only=True, map_location='cpu')
                except Exception as e1:
                    try:
                        # 如果失败，尝试使用 safetensors
                        if speaker_emb_path.endswith('.safetensors'):
                            import safetensors.torch
                            speaker_emb = safetensors.torch.load_file(speaker_emb_path)
                            if isinstance(speaker_emb, dict):
                                speaker_emb = next(iter(speaker_emb.values()))
                        else:
                            # 最后尝试使用传统方式加载
                            speaker_emb = torch.load(speaker_emb_path, weights_only=False, map_location='cpu')
                    except Exception as e2:
                        raise Exception(f"所有加载方式都失败: \n{str(e1)}\n{str(e2)}")
                
                # 验证和处理加载的数据
                if isinstance(speaker_emb, dict):
                    # 如果是字典，尝试获取第一个值
                    speaker_emb = next(iter(speaker_emb.values()))
                
                if not isinstance(speaker_emb, torch.Tensor):
                    raise ValueError(f"加载的 speaker embedding 不是 tensor 类型: {type(speaker_emb)}")
                
                # 确保维度正确
                if speaker_emb.dim() != 2 or speaker_emb.size(0) != 1:
                    speaker_emb = speaker_emb.unsqueeze(0)
                
                # 确保数据类型正确
                speaker_emb = speaker_emb.to(dtype=self.dtype)
                
            except Exception as e:
                self.logger.warning(f"加载 speaker embedding 时出错: {str(e)}")
                self.logger.warning(f"使用空的 speaker embedding")
                # 使用空的 speaker embedding 作为后备
                speaker_emb = torch.zeros((1, self.cfg.spk_emb_dim), 
                                        dtype=self.dtype, 
                                        device=self.device)
            
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

        return res_gen
