# -*- coding: utf-8 -*-
# @Time    : 2024/11/3
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: tts.py
import pdb
import random
import torch
import math
import numpy as np
import ast
import torchaudio
from omegaconf import OmegaConf
from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
from chattts_plus.commons import utils as c_utils


cfg = "../../configs/infer/chattts_plus_trt.yaml"
infer_cfg = OmegaConf.load(cfg)
pipe = ChatTTSPlusPipeline(infer_cfg, device=c_utils.get_inference_device())


def generate_audio(
        text,
        speaker_emb_path,
        **kwargs
):
    if not text:
        return None
    audio_save_path = kwargs.get("audio_save_path", None)
    params_infer_code = c_utils.InferCodeParams(
        prompt="[speed_3]",
        temperature=.0003,
        max_new_token=2048,
        top_P=0.7,
        top_K=20
    )
    params_refine_text = c_utils.RefineTextParams(
        prompt='[oral_2][laugh_0][break_4]',
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        max_new_token=384
    )
    infer_seed = kwargs.get("infer_seed", 1234)
    with c_utils.TorchSeedContext(infer_seed):
        pipe_res_gen = pipe.infer(
            text,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            use_decoder=True,
            stream=False,
            skip_refine_text=True,
            do_text_normalization=True,
            do_homophone_replacement=True,
            do_text_optimization=False,
            speaker_emb_path=speaker_emb_path,
            speaker_audio_path=None,
            speaker_audio_text=None
        )
        wavs = []
        for wavs_ in pipe_res_gen:
            wavs.extend(wavs_)
        wavs = torch.cat(wavs).cpu().float().unsqueeze(0)
        if audio_save_path:
            torchaudio.save(audio_save_path, wavs, 24000)
        return wavs


if __name__ == '__main__':
    import pickle
    import os
    from tqdm import tqdm

    base_url = "https://api2.aigcbest.top/v1"
    api_token = ""
    gpt_model = ""
    pdf_txt_file = "../../data/pdfs/AnimateAnyone.txt"
    script_pkl = os.path.splitext(pdf_txt_file)[0] + "-script.pkl"
    re_script_pkl = os.path.splitext(pdf_txt_file)[0] + "-script-rewrite.pkl"
    save_audio_path = os.path.splitext(pdf_txt_file)[0] + ".wav"
    with open(re_script_pkl, 'rb') as file:
        PODCAST_TEXT = pickle.load(file)

    final_audio = None

    i = 1
    speaker1_emb_path = "speaker_pt/en_woman_1200.pt"
    speaker2_emb_path = "speaker_pt/en_man_5200.pt"
    save_dir = os.path.splitext(pdf_txt_file)[0] + "_audios"
    os.makedirs(save_dir, exist_ok=True)
    wavs = []
    for speaker, text in tqdm(ast.literal_eval(PODCAST_TEXT), desc="Generating podcast segments", unit="segment"):
        # output_path = os.path.join(save_dir, f"_podcast_segment_{i:03d}.wav")
        output_path = None
        if speaker == "Speaker 1":
            wav_ = generate_audio(text, speaker1_emb_path, audio_save_path=output_path)
        else:
            wav_ = generate_audio(text, speaker2_emb_path, audio_save_path=output_path)
        wavs.append(wav_)
        i += 1
    wavs = torch.cat(wavs, dim=-1)
    torchaudio.save(save_audio_path, wavs, 24000)
    print(save_audio_path)
