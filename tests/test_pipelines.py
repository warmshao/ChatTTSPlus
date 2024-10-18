# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 12:48
# @Project : ChatTTSPlus
# @FileName: test_pipelines.py
import os


def test_chattts_plus_pipeline():
    import torch
    import time
    import torchaudio

    from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
    from chattts_plus.commons.utils import InferCodeParams, RefineTextParams
    from omegaconf import OmegaConf

    infer_cfg_path = "configs/infer/chattts_plus.yaml"
    infer_cfg = OmegaConf.load(infer_cfg_path)

    pipeline = ChatTTSPlusPipeline(infer_cfg, device=torch.device("cuda"))

    speaker_emb_path = "assets/speakers/2222.pt"
    params_infer_code = InferCodeParams(
        prompt="[speed_5]",
        temperature=.0003,
        max_new_token=2048,
        top_P=0.7,
        top_K=20
    )
    params_refine_text = RefineTextParams(
        prompt='[oral_2][laugh_0][break_4]',
        top_P=0.7,
        top_K=20,
        temperature=0.3,
        max_new_token=384
    )
    infer_text = "一场雨后，天空和地面互换了身份，抬头万里暗淡，足下星河生辉。这句话真是绝了.你觉得呢.哈哈哈哈"
    t0 = time.time()
    wavs = pipeline.infer(
        infer_text,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        stream=False,
        skip_refine_text=False,
        do_text_normalization=True,
        do_homophone_replacement=True,
        do_text_optimization=True,
        speaker_emb_path=None
    )
    print("total infer time:{} sec".format(time.time() - t0))
    save_dir = "results/chattts_plus"
    os.makedirs(save_dir, exist_ok=True)
    audio_save_path = f"{save_dir}/{os.path.basename(speaker_emb_path)}-{time.time()}.wav"
    torchaudio.save(audio_save_path, wavs[0].cpu().float().unsqueeze(0), 24000)
    print(audio_save_path)


if __name__ == '__main__':
    test_chattts_plus_pipeline()
