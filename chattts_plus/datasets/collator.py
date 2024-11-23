# -*- coding: utf-8 -*-
# @Time    : 2024/11/23
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: collator.py
import pdb

import torch
import torch.nn.functional as F


class BaseCollator:
    def __init__(self, text_pad: int = 0, audio_pad: int = 0):
        self.text_pad = text_pad
        self.audio_pad = audio_pad

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        audio_maxlen = max(len(item["audio_wavs"]) for item in batch)
        text_maxlen = max(len(item["text_input_ids"]) for item in batch)

        text = []
        text_input_ids = []
        text_mask = []
        audio_wavs = []
        audio_mask = []
        for x in batch:
            text.append(x["text"])
            text_input_ids.append(
                F.pad(
                    x["text_input_ids"],
                    (0, 0, text_maxlen - len(x["text_input_ids"]), 0),
                    value=self.text_pad,
                )
            )
            text_mask.append(
                F.pad(
                    x["text_mask"],
                    (text_maxlen - len(x["text_mask"]), 0),
                    value=0,
                )
            )
            audio_wavs.append(
                F.pad(
                    x["audio_wavs"],
                    (0, audio_maxlen - len(x["audio_wavs"])),
                    value=self.audio_pad,
                )
            )
            audio_mask.append(
                F.pad(
                    x["audio_mask"],
                    (0, audio_maxlen - len(x["audio_mask"])),
                    value=0,
                )
            )
        return {
            "text": text,
            "text_input_ids": torch.stack(text_input_ids),
            "text_mask": torch.stack(text_mask),
            "audio_wavs": torch.stack(audio_wavs),
            "audio_mask": torch.stack(audio_mask),
        }
