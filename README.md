## ChatTTSPlus: Extension of ChatTTS

<a href="README_ZH.md">中文</a> | <a href="README.md">English</a>

ChatTTSPlus is an extension of [ChatTTS](https://github.com/2noise/ChatTTS), adding features such as TensorRT acceleration, voice cloning, and mobile model deployment.

**If you find this project useful, please give it a star! ✨✨**

### Some fun demos based on ChatTTSPlus
* NotebookLM podcast: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jz8HPdRe_igoNjMSv0RaTn3l2c3seYFT?usp=sharing).
Use ChatTTSPlus to turn the `AnimateAnyone` paper into a podcast.

   <video src="https://github.com/user-attachments/assets/82afa5de-1bf2-404a-ab10-06d52b16a8f9" controls="controls" width="300" height="500">Your browser does not support playing this video!</video>

### New Features
- [x] Refactored ChatTTS code in a way I'm familiar with.
- [x] **Achieved over 3x acceleration with TensorRT**, increasing performance on a Windows 3060 GPU from 28 tokens/s to 110 tokens/s.
- [x] Windows integration package for one-click extraction and use.
- [x] Implemented voice cloning using technologies like LoRA. Please reference [voice_clone](assets/docs/voice_clone.md).
- [ ] Model compression and acceleration using techniques like pruning and knowledge distillation, targeting mobile deployment.

### Environment Setup
* Install Python 3; it's recommended to use [Miniforge](https://github.com/conda-forge/miniforge). Run: `conda create -n chattts_plus python=3.10 && conda activate chattts_plus`
* Download the source code: `git clone https://github.com/warmshao/ChatTTSPlus`, and navigate to the project root directory: `cd ChatTTSPlus`
* Install necessary Python libraries: `pip install -r requirements.txt`
* [Optional] If you want to use TensorRT, please install [tensorrt10](https://developer.nvidia.com/tensorrt/download)
* [Recommended for Windows users] Download the integration package directly from [Google Drive Link](https://drive.google.com/file/d/1yOnU5dRTJvFnc4wyw02nAeJH5_FgNod2/view?usp=sharing), extract it, and double-click `webui.bat` to use. If you want to update the code, please double-click `update.bat`. Note: **This will overwrite all your local code modifications.**

### Demo
* Web UI with TensorRT: `python webui.py --cfg configs/infer/chattts_plus_trt.yaml`. 
* Web UI with PyTorch: `python webui.py --cfg configs/infer/chattts_plus.yaml`

<video src="https://github.com/user-attachments/assets/bd2c1e48-6339-4ad7-bcfa-ed008c992594" controls="controls" width="500" height="300">Your browser does not support playing this video!</video>

### License
ChatTTSPlus inherits the license from [ChatTTS](https://github.com/2noise/ChatTTS); please refer to [ChatTTS](https://github.com/2noise/ChatTTS) as the standard.

The code is published under the AGPLv3+ license.

The model is published under the CC BY-NC 4.0 license. It is intended for educational and research use and should not be used for any commercial or illegal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data used in this repository are for academic and research purposes only. The data is obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.

### Disclaimer
We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

### About Me
I'm an algorithm engineer focused on implementing AIGC and LLM-related products. If you have any needs for entrepreneurship, collaboration, or customization, feel free to add me on Discord (warmshao) or on WeChat:

<img src="assets/wx/shipinhao.jpg" alt="微信" width="300" height="340">
