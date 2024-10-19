# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 15:53
# @Project : ChatTTSPlus
# @FileName: test_models.py
import os
import pdb
import torch
import torchaudio


def test_tokenizer():
    from chattts_plus.models.tokenizer import Tokenizer
    model_path = "checkpoints/asset/tokenizer.pt"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        weight_type = torch.float16
    else:
        device = torch.device("cpu")
        weight_type = torch.float32
    tokenizer_ = Tokenizer(model_path)

    text = "hello world!"
    input_ids, attention_mask, text_mask = tokenizer_.encode([text], 4, None, device)


def test_dvae_encode():
    from chattts_plus.models.dvae import DVAE
    if torch.cuda.is_available():
        device = torch.device("cuda")
        weight_type = torch.float16
    else:
        device = torch.device("cpu")
        weight_type = torch.float32

    audio_file = "data/xionger/slicer_opt/vocal_5.WAV_10.wav_0000251200_0000423680.wav"
    audio_wav, audio_sr_ = torchaudio.load(audio_file)
    audio_sr = 24000
    audio_wav = torchaudio.functional.resample(audio_wav, orig_freq=audio_sr_, new_freq=audio_sr)
    audio_wav = torch.mean(audio_wav, 0).to(device, dtype=weight_type)

    decoder_config = dict(
        idim=512,
        odim=512,
        hidden=256,
        n_layer=12,
        bn_dim=128,
    )
    encoder_config = dict(
        idim=512,
        odim=1024,
        hidden=256,
        n_layer=12,
        bn_dim=128,
    )
    vq_config = dict(
        dim=1024,
        levels=(5, 5, 5, 5),
        G=2,
        R=2,
    )
    model_path = "checkpoints/asset/DVAE_full.pt"
    dvae_encoder = DVAE(
        decoder_config=decoder_config,
        encoder_config=encoder_config,
        vq_config=vq_config,
        dim=decoder_config["idim"],
        model_path=model_path,
        coef=None,
    )
    dvae_encoder = dvae_encoder.eval().to(device, dtype=weight_type)
    audio_ids = dvae_encoder(audio_wav, "encode")
    pdb.set_trace()


def test_dvae_decode():
    from chattts_plus.models.dvae import DVAE
    if torch.cuda.is_available():
        device = torch.device("cuda")
        weight_type = torch.float16
    else:
        device = torch.device("cpu")
        weight_type = torch.float32
    decoder_config = dict(
        idim=384,
        odim=384,
        hidden=512,
        n_layer=12,
        bn_dim=128
    )
    model_path = "checkpoints/asset/Decoder.pt"
    dvae_decoder = DVAE(
        decoder_config=decoder_config,
        dim=decoder_config["idim"],
        coef=None,
        model_path=model_path
    )
    dvae_decoder = dvae_decoder.eval().to(device, dtype=weight_type)

    vq_feats = torch.randn(1, 768, 388).to(device, dtype=weight_type)
    mel_feats = dvae_decoder(vq_feats)
    pdb.set_trace()


def test_vocos():
    import vocos
    import vocos.feature_extractors
    import vocos.models
    import vocos.heads

    feature_extractor_cfg = dict(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
    )
    backbone_cfg = dict(
        input_channels=100,
        dim=512,
        intermediate_dim=1536,
        num_layers=8
    )
    head_cfg = dict(
        dim=512,
        n_fft=1024,
        hop_length=256,
        padding="center"
    )
    feature_extractor = vocos.feature_extractors.MelSpectrogramFeatures(**feature_extractor_cfg)
    backbone = vocos.models.VocosBackbone(**backbone_cfg)
    head = vocos.heads.ISTFTHead(**head_cfg)

    device = torch.device("cuda")
    dtype = torch.float16
    if "mps" in str(device):
        device = torch.device("cpu")
        dtype = torch.float32

    vocos = vocos.Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head).to(
        device, dtype=dtype).eval()
    vocos_ckpt_path = "checkpoints/asset/Vocos.pt"
    vocos.load_state_dict(torch.load(vocos_ckpt_path, weights_only=True, mmap=True))

    mel_feats = torch.randn(1, 100, 388 * 2).to(device, dtype=dtype)
    audio_wavs = vocos.decode(mel_feats).cpu().float()
    result_dir = "./results/test_vocos"
    os.makedirs(result_dir, exist_ok=True)
    torchaudio.save(os.path.join(result_dir, "test.wav"), audio_wavs, sample_rate=24000)


def test_gpt():
    from chattts_plus.models.gpt import GPT

    if torch.cuda.is_available():
        device = torch.device("cuda")
        weight_type = torch.float16
    else:
        device = torch.device("cpu")
        weight_type = torch.float32
    model_path = "checkpoints/asset/GPT.pt"
    gpt_cfg = dict(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=20,
        use_cache=False,
        max_position_embeddings=4096,
        spk_emb_dim=192,
        spk_KL=False,
        num_audio_tokens=626,
        num_vq=4,
    )
    gpt = GPT(gpt_cfg, model_path=model_path)

    gpt = gpt.eval().to(device, dtype=weight_type)
    pdb.set_trace()


if __name__ == '__main__':
    # test_tokenizer()
    test_dvae_encode()
    # test_dvae_decode()
    # test_vocos()
    # test_gpt()
