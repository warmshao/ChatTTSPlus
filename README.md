## ChatTTSPlus: Extension of ChatTTS

<a href="README.md">English</a> | <a href="README_CN.md">中文</a>

ChatTTSPlus is an extension of [ChatTTS](https://github.com/2noise/ChatTTS), adding features like TensorRT acceleration, voice cloning, and mobile device compatibility.

### New Features
- [x] TensorRT acceleration, achieving a speed increase from 28 tokens/s to 110 tokens/s on a Windows 3060 GPU.
- [ ] Added API and command line usage options.
- [ ] Windows integration package for one-click extraction and use.
- [ ] Voice cloning using techniques like LoRA.
- [ ] Model compression and acceleration using pruning and knowledge distillation, with the goal of running on mobile devices.

### Environment Setup
* Install Python 3, preferably with [miniforge](https://github.com/conda-forge/miniforge): `conda create -n flip python=3.10 && conda activate chattts_plus`
* Install required Python libraries: `pip install -r requirements.txt`
* **Optional**: If you plan to use TensorRT, install it with: `pip install --pre --extra-index-url https://pypi.nvidia.com/ tensorrt --no-cache-dir`

### Demo
<video src="https://github.com/user-attachments/assets/bd2c1e48-6339-4ad7-bcfa-ed008c992594" controls="controls" width="500" height="300">Your browser does not support video playback!</video>

### License
ChatTTSPlus inherits the license from [ChatTTS](https://github.com/2noise/ChatTTS); please refer to the [ChatTTS](https://github.com/2noise/ChatTTS) license as the standard.

The code is published under the AGPLv3+ license.

The model is published under the CC BY-NC 4.0 license. It is intended for educational and research use only, not for commercial or illegal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data in this repository are for academic and research purposes only. All data is obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.
