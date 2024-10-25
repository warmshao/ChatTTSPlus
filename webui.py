import os, sys
import pdb

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import random
import argparse
import gradio as gr
from omegaconf import OmegaConf
import numpy as np
import math
import torch
from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
from chattts_plus.commons import utils
from chattts_plus.commons import constants

# ChatTTSPlus pipeline
pipe: ChatTTSPlusPipeline = None

js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

seed_min = 1
seed_max = 4294967295


def generate_seed():
    return gr.update(value=random.randint(seed_min, seed_max))


def update_spk_emb_path(file):
    spk_emb_path = file.name
    return spk_emb_path


def list_pt_files_in_dir(directory):
    if os.path.isdir(directory):
        pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
        return gr.Dropdown(label="Select Speaker Embedding", choices=pt_files) if pt_files else gr.Dropdown(
            label="Select Speaker Embedding", choices=[])
    return gr.Dropdown(label="Select Speaker Embedding", choices=[])


def set_spk_emb_path_from_dir(directory, selected_file):
    spk_emb_path = os.path.join(directory, selected_file)
    return spk_emb_path


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def refine_text(
        text,
        prompt,
        temperature,
        top_P,
        top_K,
        text_seed_input,
        refine_text_flag,
):
    global pipe

    if not refine_text_flag:
        sleep(1)  # to skip fast answer of loading mark
        return text

    with utils.TorchSeedContext(text_seed_input):
        params_refine_text = utils.RefineTextParams(
            prompt=prompt,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            max_new_token=384
        )
        text_gen = pipe.infer(
            text,
            skip_refine_text=False,
            refine_text_only=True,
            do_text_normalization=True,
            do_homophone_replacement=True,
            do_text_optimization=True,
            params_refine_text=params_refine_text
        )
        texts = []
        for text_ in text_gen:
            texts.extend(text_)

    return "".join(texts)


def generate_audio(
        text,
        prompt,
        temperature,
        top_P,
        top_K,
        spk_emb_path,
        stream,
        audio_seed_input,
        sample_text_input,
        sample_audio_input,
        activate_tag_name
):
    global pipe

    if not text:
        return None

    params_infer_code = utils.InferCodeParams(
        prompt=prompt,
        temperature=temperature,
        top_P=top_P,
        top_K=top_K,
        max_new_token=2048,
    )
    if activate_tag_name == "Speaker Embedding":
        sample_text_input = None
        sample_audio_input = None

    with utils.TorchSeedContext(audio_seed_input):
        wav_gen = pipe.infer(
            text,
            skip_refine_text=True,
            do_text_normalization=False,
            do_homophone_replacement=False,
            do_text_optimization=False,
            params_infer_code=params_infer_code,
            stream=stream,
            speaker_audio_path=sample_text_input,
            speaker_audio_text=sample_audio_input,
            speaker_emb_path=spk_emb_path
        )
        if stream:
            for gen in wav_gen:
                audio = gen[0].cpu().float().numpy()
                if audio is not None and len(audio) > 0:
                    yield 24000, float_to_int16(audio).T
        else:
            wavs = []
            for wavs_ in wav_gen:
                wavs.extend(wavs_)
            wavs = torch.cat(wavs).cpu().float().numpy()
            yield 24000, float_to_int16(wavs).T


def update_active_tab(tab_name):
    return gr.State(tab_name)


def main(args):
    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), js=js_func) as demo:
        gr.Markdown("# ChatTTSPlus WebUI")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    max_lines=4,
                    placeholder="Please Input Text...",
                    interactive=True,
                )
        activate_tag_name = gr.State(value="Speaker Embedding")

        with gr.Tabs() as tabs:
            with gr.Tab("Speaker Embedding"):
                with gr.Column():
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            spk_emb_dir = gr.Textbox(label="Input Speaker Embedding Directory",
                                                     placeholder="Please input speaker embedding directory",
                                                     value=os.path.abspath(
                                                         os.path.join(constants.PROJECT_DIR, "assets/speakers")))
                            reload_chat_button = gr.Button("Reload", scale=1)
                            pt_files_dropdown = gr.Dropdown(label="Select Speaker Embedding")

                        upload_emb_file = gr.File(label="Upload Speaker Embedding File (.pt)")

                    spk_emb_path = gr.Textbox(
                        label="Speaker Embedding Path",
                        max_lines=3,
                        show_copy_button=True,
                        interactive=True,
                        scale=2,
                    )

                    upload_emb_file.upload(update_spk_emb_path, inputs=upload_emb_file, outputs=spk_emb_path)
                    reload_chat_button.click(
                        list_pt_files_in_dir, inputs=spk_emb_dir, outputs=pt_files_dropdown
                    )
                    pt_files_dropdown.change(
                        set_spk_emb_path_from_dir, inputs=[spk_emb_dir, pt_files_dropdown], outputs=spk_emb_path
                    )

            with gr.Tab("Speaker Audio (ZeroShot)"):
                with gr.Row(equal_height=True):
                    sample_audio_input = gr.Audio(
                        value=None,
                        type="filepath",
                        interactive=True,
                        show_label=False,
                        waveform_options=gr.WaveformOptions(
                            sample_rate=24000,
                        ),
                    )
                    sample_text_input = gr.Textbox(
                        label="Sample Text (ZeroShot)",
                        lines=4,
                        max_lines=4,
                        placeholder="If Sample Audio and Sample Text are available, the Speaker Embedding will be disabled.",
                        interactive=True,
                    )

        tabs.change(
            fn=update_active_tab,
            inputs=tabs,
            outputs=[activate_tag_name]
        )

        with gr.Row(equal_height=True):
            refine_text_checkbox = gr.Checkbox(
                label="Refine text", interactive=True, value=True
            )
            text_prompt = gr.Text(
                interactive=True,
                value="[oral_2][laugh_0][break_4]",
                label="text_prompt"
            )
            text_temperature_slider = gr.Number(
                minimum=0.00001,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="Text Temperature",
                interactive=True,
            )
            text_top_p_slider = gr.Number(
                minimum=0.1,
                maximum=0.9,
                value=0.7,
                step=0.05,
                label="text_top_P",
                interactive=True,
            )
            text_top_k_slider = gr.Number(
                minimum=1,
                maximum=30,
                value=20,
                step=1,
                label="text_top_K",
                interactive=True,
            )
            text_seed_input = gr.Number(
                label="Text Seed",
                interactive=True,
                value=1,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_text_seed = gr.Button("\U0001F3B2", interactive=True)

        with gr.Row(equal_height=True):
            audio_prompt = gr.Text(
                interactive=True,
                value="[speed_5]",
                label="audio_prompt"
            )
            audio_temperature_slider = gr.Number(
                minimum=0.00001,
                maximum=1.0,
                step=0.0001,
                value=0.0003,
                label="Audio Temperature",
                interactive=True,
            )
            audio_top_p_slider = gr.Number(
                minimum=0.1,
                maximum=0.9,
                value=0.7,
                step=0.05,
                label="audio_top_P",
                interactive=True,
            )
            audio_top_k_slider = gr.Number(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="audio_top_K",
                interactive=True,
            )
            audio_seed_input = gr.Number(
                label="Audio Seed",
                interactive=True,
                value=1,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_audio_seed = gr.Button("\U0001F3B2", interactive=True)

        with gr.Row():
            auto_play_checkbox = gr.Checkbox(
                label="Auto Play", value=False, scale=1, interactive=True
            )
            stream_mode_checkbox = gr.Checkbox(
                label="Stream Mode",
                value=False,
                scale=1,
                interactive=True,
            )
            generate_button = gr.Button(
                "Generate", scale=2, variant="primary", interactive=True
            )
            interrupt_button = gr.Button(
                "Interrupt",
                scale=2,
                variant="stop",
                visible=False,
                interactive=False,
            )

        text_output = gr.Textbox(
            label="Output Text",
            interactive=False,
            show_copy_button=True,
            lines=4,
        )

        generate_audio_seed.click(generate_seed, outputs=audio_seed_input)
        generate_text_seed.click(generate_seed, outputs=text_seed_input)

        @gr.render(inputs=[auto_play_checkbox, stream_mode_checkbox])
        def make_audio(autoplay, stream):
            audio_output = gr.Audio(
                label="Output Audio",
                value=None,
                format="mp3" if not stream else "wav",
                autoplay=autoplay,
                streaming=stream,
                interactive=False,
                show_label=True,
                waveform_options=gr.WaveformOptions(
                    sample_rate=24000,
                ),
            )

            generate_button.click(
                fn=refine_text,
                inputs=[
                    text_input,
                    text_prompt,
                    text_temperature_slider,
                    text_top_p_slider,
                    text_top_k_slider,
                    text_seed_input,
                    refine_text_checkbox,
                ],
                outputs=text_output,
            ).then(
                generate_audio,
                inputs=[
                    text_output,
                    audio_prompt,
                    audio_temperature_slider,
                    audio_top_p_slider,
                    audio_top_k_slider,
                    spk_emb_path,
                    stream_mode_checkbox,
                    audio_seed_input,
                    sample_text_input,
                    sample_audio_input,
                    activate_tag_name
                ],
                outputs=audio_output,
            )

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=True,
        show_api=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument(
        "--cfg", type=str, default="configs/infer/chattts_plus.yaml", help="config of chattts plus"
    )
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="server name"
    )
    parser.add_argument("--server_port", type=int, default=7890, help="server port")
    args = parser.parse_args()

    infer_cfg = OmegaConf.load(args.cfg)
    pipe = ChatTTSPlusPipeline(infer_cfg, device=utils.get_inference_device())
    main(args)
