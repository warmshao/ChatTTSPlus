# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 12:48
# @Project : ChatTTSPlus
# @FileName: test_pipelines.py

def test_chattts_plus_pipeline():
    import torch
    from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
    from omegaconf import OmegaConf

    infer_cfg_path = "configs/infer/chattts_plus.yaml"
    infer_cfg = OmegaConf.load(infer_cfg_path)

    pipeline = ChatTTSPlusPipeline(infer_cfg, device=torch.device("cpu"))


if __name__ == '__main__':
    test_chattts_plus_pipeline()
