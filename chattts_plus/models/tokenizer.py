import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
"""

from typing import List, Tuple, Optional
import lzma

import numpy as np
import pybase16384 as b14
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

from ..commons import logger


class Tokenizer:
    def __init__(
            self, model_path, **kwargs
    ):
        self.logger = logger.get_logger(self.__class__.__name__)
        self.logger.info(f"loading Tokenizer pretrained model: {model_path}")

        tokenizer: BertTokenizerFast = torch.load(
            model_path, map_location="cpu", mmap=True
        )
        self._tokenizer = tokenizer

        # 设置特殊token
        self._tokenizer.eos_token = '[SEP]'  # 使用BERT的默认结束符
        self._tokenizer.pad_token = '[PAD]'  # 使用BERT的默认填充符
        
        # 获取或设置对应的token id
        if not hasattr(self._tokenizer, 'pad_token_id'):
            self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids('[PAD]')
        if not hasattr(self._tokenizer, 'eos_token_id'):    
            self._tokenizer.eos_token_id = self._tokenizer.convert_tokens_to_ids('[SEP]')

        self.len = len(tokenizer)
        self.spk_emb_ids = tokenizer.convert_tokens_to_ids("[spk_emb]")
        self.break_0_ids = tokenizer.convert_tokens_to_ids("[break_0]")
        self.eos_token = tokenizer.convert_tokens_to_ids("[Ebreak]")

        self.decode = self._tokenizer.batch_decode

    @torch.inference_mode()
    def encode(
            self,
            text: List[str],
            num_vq: int,
            prompt_str: Optional[str] = None,
            device="cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_ids_lst = []
        attention_mask_lst = []
        max_input_ids_len = -1
        max_attention_mask_len = -1
        prompt_size = 0

        prompt = self._decode_prompt(prompt_str) if prompt_str is not None else None

        if prompt is not None:
            assert prompt.size(0) == num_vq, "prompt dim 0 must equal to num_vq"
            prompt_size = prompt.size(1)

        # avoid random speaker embedding of tokenizer in the other dims
        for t in text:
            x = self._tokenizer.encode_plus(
                t, return_tensors="pt", add_special_tokens=False, padding=True
            )
            input_ids_lst.append(x["input_ids"].squeeze_(0))
            attention_mask_lst.append(x["attention_mask"].squeeze_(0))
            ids_sz = input_ids_lst[-1].size(0)
            if ids_sz > max_input_ids_len:
                max_input_ids_len = ids_sz
            attn_sz = attention_mask_lst[-1].size(0)
            if attn_sz > max_attention_mask_len:
                max_attention_mask_len = attn_sz

        if prompt is not None:
            max_input_ids_len += prompt_size
            max_attention_mask_len += prompt_size

        input_ids = torch.zeros(
            len(input_ids_lst),
            max_input_ids_len,
            device=device,
            dtype=input_ids_lst[0].dtype,
        )
        for i in range(len(input_ids_lst)):
            input_ids.narrow(0, i, 1).narrow(
                1,
                max_input_ids_len - prompt_size - input_ids_lst[i].size(0),
                input_ids_lst[i].size(0),
            ).copy_(
                input_ids_lst[i]
            )  # left padding

        attention_mask = torch.zeros(
            len(attention_mask_lst),
            max_attention_mask_len,
            device=device,
            dtype=attention_mask_lst[0].dtype,
        )
        for i in range(len(attention_mask_lst)):
            attn = attention_mask.narrow(0, i, 1)
            attn.narrow(
                1,
                max_attention_mask_len - prompt_size - attention_mask_lst[i].size(0),
                attention_mask_lst[i].size(0),
            ).copy_(
                attention_mask_lst[i]
            )  # left padding
            if prompt_size > 0:
                attn.narrow(
                    1,
                    max_attention_mask_len - prompt_size,
                    prompt_size,
                ).fill_(1)

        text_mask = attention_mask.bool()
        new_input_ids = input_ids.unsqueeze_(-1).expand(-1, -1, num_vq).clone()

        if prompt_size > 0:
            text_mask.narrow(1, max_input_ids_len - prompt_size, prompt_size).fill_(0)
            prompt_t = prompt.t().unsqueeze_(0).expand(new_input_ids.size(0), -1, -1)
            new_input_ids.narrow(
                1,
                max_input_ids_len - prompt_size,
                prompt_size,
            ).copy_(prompt_t)

        return new_input_ids, attention_mask, text_mask

    @staticmethod
    def _decode_spk_emb(spk_emb: str) -> np.ndarray:
        return np.frombuffer(
            lzma.decompress(
                b14.decode_from_string(spk_emb),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
            dtype=np.float16,
        ).copy()

    @torch.no_grad()
    def apply_spk_emb(
            self,
            emb: torch.Tensor,
            spk_emb,
            input_ids: torch.Tensor,
            device: torch.device,
    ):
        if isinstance(spk_emb, str):
            spk_emb_tensor = torch.from_numpy(self._decode_spk_emb(spk_emb))
        else:
            spk_emb_tensor = spk_emb

        n = (
            F.normalize(
                spk_emb_tensor,
                p=2.0,
                dim=0,
                eps=1e-12,
            )
            .to(emb.device, dtype=emb.dtype)
            .unsqueeze_(0)
            .expand(emb.size(0), -1)
            .unsqueeze_(1)
            .expand(emb.shape)
        )
        cond = input_ids.narrow(-1, 0, 1).eq(self.spk_emb_ids).expand(emb.shape)
        torch.where(cond, n, emb, out=emb)
        return emb

    @staticmethod
    @torch.no_grad()
    def _decode_prompt(prompt: str) -> torch.Tensor:
        dec = b14.decode_from_string(prompt)
        shp = np.frombuffer(dec[:4], dtype="<u2")
        p = np.frombuffer(
            lzma.decompress(
                dec[4:],
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
            dtype="<u2",
        ).copy()
        return torch.from_numpy(p).view(*shp)

    @staticmethod
    @torch.no_grad()
    def _encode_prompt(prompt: torch.Tensor) -> str:
        arr: np.ndarray = prompt.to(dtype=torch.uint16, device="cpu").numpy()
        shp = arr.shape
        assert len(shp) == 2, "prompt must be a 2D tensor"
        s = b14.encode_to_string(
            np.array(shp, dtype="<u2").tobytes()
            + lzma.compress(
                arr.astype("<u2").tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
        )
        return s

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
