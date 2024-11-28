# -*- coding: utf-8 -*-
# @Time    : 2024/11/23
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: base_dataset.py

import os.path
import pdb

import torch
import torchaudio
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    设置一个通用的 train dataset loader
    """

    def __init__(self, meta_infos=[],
                 tokenizer=None,
                 normalizer=None,
                 sample_rate=24_000,
                 num_vq=4,
                 use_empty_speaker=False,
                 **kwargs
                 ):
        super(BaseDataset, self).__init__()
        self.meta_infos = meta_infos
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.sample_rate = sample_rate
        self.num_vq = num_vq
        self.use_empty_speaker = use_empty_speaker
        self.data_infos, self.speakers = self.load_data()

    def load_data(self, **kwargs):
        data_infos = []
        speakers = set()
        for info_path in self.meta_infos:
            data_root = os.path.dirname(info_path)
            with open(info_path, "r", encoding='UTF-8') as fin:
                for line in fin.readlines():
                    line_splits = line.strip().replace('\n', '').split("|")
                    if len(line_splits) == 4:
                        speakers.add(line_splits[0])
                        data_infos.append(
                            {
                                "speaker": line_splits[0],
                                "audio_path": line_splits[1],
                                "text": line_splits[-1],
                                "lang": line_splits[2].lower()
                            }
                        )
        return data_infos, speakers

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, i):
        data_info_ = self.data_infos[i]
        audio_wavs, audio_mask = self.preprocess_audio(data_info_["audio_path"])
        text_input_ids, text_mask = self.preprocess_text(data_info_["text"], data_info_["lang"])
        return {
            "speaker": data_info_["speaker"],
            "text": data_info_["text"],
            "audio_wavs": audio_wavs,
            "audio_mask": audio_mask,
            "text_input_ids": text_input_ids,
            "text_mask": text_mask
        }

    def preprocess_text(
            self,
            text,
            lang="zh",
            do_text_normalization=True,
            do_homophone_replacement=True,
    ):

        text = self.normalizer(
            text,
            do_text_normalization,
            do_homophone_replacement,
            lang,
        )
        if self.use_empty_speaker:
            text = f'[Stts][empty_spk]{text}[Ptts]'
        else:
            text = f'[Stts][spk_emb]{text}[Ptts]'
        input_ids, attention_mask, text_mask = self.tokenizer.encode([text], num_vq=self.num_vq)
        return input_ids.squeeze(0), text_mask.squeeze(0)

    def preprocess_audio(self, audio_path):
        audio_wavs, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sample_rate:
            audio_wavs = torchaudio.functional.resample(
                audio_wavs,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
        audio_wavs = audio_wavs.mean(0)
        audio_mask = torch.ones(len(audio_wavs))
        return audio_wavs, audio_mask
