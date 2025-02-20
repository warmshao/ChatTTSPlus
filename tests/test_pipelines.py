# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 12:48
# @Project : ChatTTSPlus
# @FileName: test_pipelines.py
import os
import pdb


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

    params_infer_code = InferCodeParams(
        prompt="[speed_3]",
        temperature=.0003,
        max_new_token=2048,
        top_P=0.7,
        top_K=20
    )
    params_refine_text = RefineTextParams(
        prompt='[oral_2][laugh_3][break_4]',
        top_P=0.7,
        top_K=20,
        temperature=0.7,
        max_new_token=384
    )
    infer_text = ["我们针对对话式任务进行了优化，能够实现自然且富有表现力的合成语音"]
    t0 = time.time()
    # leijun: outputs/leijun_lora-1732802535.8597126/checkpoints/step-2000
    # xionger: outputs/xionger_lora-1732809910.2932503/checkpoints/step-600
    lora_path = "outputs/leijun_lora-1734532984.1128285/checkpoints/step-2000"
    pipe_res_gen = pipeline.infer(
        infer_text,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        stream=False,
        skip_refine_text=True,
        do_text_normalization=True,
        do_homophone_replacement=True,
        do_text_optimization=True,
        lora_path=lora_path,
        speaker_emb_path=''
    )
    wavs = []
    for wavs_ in pipe_res_gen:
        wavs.extend(wavs_)
    print("total infer time:{} sec".format(time.time() - t0))
    save_dir = "results/chattts_plus"
    os.makedirs(save_dir, exist_ok=True)
    audio_save_path = f"{save_dir}/{os.path.basename(lora_path)}-{time.time()}.wav"
    torchaudio.save(audio_save_path, torch.cat(wavs).cpu().float().unsqueeze(0), 24000)
    print(audio_save_path)


def test_chattts_plus_trt_pipeline():
    import torch
    import time
    import torchaudio

    from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
    from chattts_plus.commons.utils import InferCodeParams, RefineTextParams
    from omegaconf import OmegaConf

    infer_cfg_path = "configs/infer/chattts_plus_trt.yaml"
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
    infer_text = [
        "一场雨后，天空和地面互换了身份，抬头万里暗淡，足下星河生辉。这句话真是绝了.你觉得呢.哈哈哈哈",
        "本邮件内容是根据招商银行客户提供的个人邮箱发送给其本人的电子邮件，如您并非抬头标明的收件人，请您即刻删除本邮件，勿以任何形式使用及传播本邮件内容，谢谢！"
    ]
    t0 = time.time()
    pipe_res_gen = pipeline.infer(
        infer_text,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        stream=False,
        skip_refine_text=False,
        do_text_normalization=True,
        do_homophone_replacement=True,
        do_text_optimization=True,
        speaker_emb_path=speaker_emb_path
    )
    wavs = []
    for wavs_ in pipe_res_gen:
        wavs.extend(wavs_)
    print("total infer time:{} sec".format(time.time() - t0))
    save_dir = "results/chattts_plus"
    os.makedirs(save_dir, exist_ok=True)
    audio_save_path = f"{save_dir}/{os.path.basename(speaker_emb_path)}-{time.time()}.wav"
    torchaudio.save(audio_save_path, torch.cat(wavs).cpu().float().unsqueeze(0), 24000)
    print(audio_save_path)


def test_chattts_plus_zero_shot_pipeline():
    import torch
    import time
    import torchaudio

    from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
    from chattts_plus.commons.utils import InferCodeParams, RefineTextParams
    from omegaconf import OmegaConf

    infer_cfg_path = "configs/infer/chattts_plus.yaml"
    infer_cfg = OmegaConf.load(infer_cfg_path)

    pipeline = ChatTTSPlusPipeline(infer_cfg, device=torch.device("cuda"))

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
    infer_text = [
        "一场雨后，天空和地面互换了身份，抬头万里暗淡，足下星河生辉。这句话真是绝了.你觉得呢.哈哈哈哈",
        "本邮件内容是根据招商银行客户提供的个人邮箱发送给其本人的电子邮件，如您并非抬头标明的收件人，请您即刻删除本邮件，勿以任何形式使用及传播本邮件内容，谢谢！"
    ]
    speaker_audio_path = "data/xionger/slicer_opt/vocal_1.WAV_10.wav_0000000000_0000152640.wav"
    speaker_audio_text = "嘿嘿，最近我看了寄生虫，真的很推荐哦。"
    t0 = time.time()
    pipe_res_gen = pipeline.infer(
        infer_text,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        stream=False,
        skip_refine_text=False,
        do_text_normalization=True,
        do_homophone_replacement=True,
        do_text_optimization=True,
        speaker_emb_path=None,
        speaker_audio_path=speaker_audio_path,
        speaker_audio_text=speaker_audio_text
    )
    wavs = []
    for wavs_ in pipe_res_gen:
        wavs.extend(wavs_)
    print("total infer time:{} sec".format(time.time() - t0))
    save_dir = "results/chattts_plus"
    os.makedirs(save_dir, exist_ok=True)
    audio_save_path = f"{save_dir}/{os.path.basename(speaker_audio_path)}-{time.time()}.wav"
    torchaudio.save(audio_save_path, torch.cat(wavs).cpu().float().unsqueeze(0), 24000)
    print(audio_save_path)


def test_chattts_plus_zero_shot_trt_pipeline():
    import torch
    import time
    import torchaudio

    from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
    from chattts_plus.commons.utils import InferCodeParams, RefineTextParams
    from omegaconf import OmegaConf

    infer_cfg_path = "configs/infer/chattts_plus_trt.yaml"
    infer_cfg = OmegaConf.load(infer_cfg_path)

    pipeline = ChatTTSPlusPipeline(infer_cfg, device=torch.device("cuda"))

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
    infer_text = [
        "请您即刻删除本邮件，勿以任何形式使用及传播本邮件内容，谢谢！"
    ]
    speaker_audio_path = "data/xionger/slicer_opt/vocal_1.WAV_10.wav_0000000000_0000152640.wav"
    speaker_audio_text = "嘿嘿，最近我看了寄生虫，真的很推荐哦。"
    t0 = time.time()
    pipe_res_gen = pipeline.infer(
        infer_text,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        stream=False,
        skip_refine_text=False,
        do_text_normalization=True,
        do_homophone_replacement=True,
        do_text_optimization=True,
        speaker_emb_path=None,
        speaker_audio_path=speaker_audio_path,
        speaker_audio_text=speaker_audio_text
    )
    wavs = []
    for wavs_ in pipe_res_gen:
        wavs.extend(wavs_)
    print("total infer time:{} sec".format(time.time() - t0))
    save_dir = "results/chattts_plus"
    os.makedirs(save_dir, exist_ok=True)
    audio_save_path = f"{save_dir}/{os.path.basename(speaker_audio_path)}-{time.time()}.wav"
    torchaudio.save(audio_save_path, torch.cat(wavs).cpu().float().unsqueeze(0), 24000)
    print(audio_save_path)


if __name__ == '__main__':
    test_chattts_plus_pipeline()
    # test_chattts_plus_trt_pipeline()
    # test_chattts_plus_zero_shot_pipeline()
    # test_chattts_plus_zero_shot_trt_pipeline()
