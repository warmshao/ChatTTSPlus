## ChatTTSPlus: Extension of ChatTTS

<a href="README_ZH.md">中文</a> | <a href="README.md">English</a>

ChatTTSPlus是[ChatTTS](https://github.com/2noise/ChatTTS)的扩展，增加使用TensorRT加速、声音克隆和模型移动端运行等功能。

**如果你觉得这个项目有用，帮我点个star吧✨✨**

### 基于ChatTTSPlus做的有趣的demo
* NotebookLM播客: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jz8HPdRe_igoNjMSv0RaTn3l2c3seYFT?usp=sharing)
，使用ChatTTS把 `AnimateAnyone` 这篇文章变成播客。

   <video src="https://github.com/user-attachments/assets/82afa5de-1bf2-404a-ab10-06d52b16a8f9" controls="controls" width="300" height="500">Your browser does not support playing this video!</video>

### 新增功能
- [x] 将ChatTTS的代码以我熟悉的方式重构。
- [x] **使用TensorRT实现3倍以上的加速**, 在windows的3060显卡上从28token/s提升到110token/s。
- [x] windows整合包，一键解压使用。
- [x] 使用Lora等技术实现声音克隆。请参考 [声音克隆](assets/docs/voice_clone_ZH.md)
- [ ] 使用剪枝、知识蒸馏等做模型压缩和加速，目标在移动端运行。

### 环境安装
* 安装python3，推荐可以用[miniforge](https://github.com/conda-forge/miniforge).`conda create -n chattts_plus python=3.10 && conda activate chattts_plus`
* 下载源码: `git clone https://github.com/warmshao/ChatTTSPlus`, 并到项目根目录下: `cd ChatTTSPlus`
* 安装必要的python库, `pip install -r requirements.txt`
* 【可选】如果你要使用tensorrt的话，请安装[tensorrt10](https://developer.nvidia.com/tensorrt/download)
* 【windows用户推荐】直接从[Google Drive链接](https://drive.google.com/file/d/1yOnU5dRTJvFnc4wyw02nAeJH5_FgNod2/view?usp=sharing)下载整合包，解压后双击`webui.bat`即可使用。如果要更新代码的话，请先双击`update.bat`, 注意：**这会覆盖你本地所有的代码修改**。

### Demo
* Webui with TensorRT: `python webui.py --cfg configs/infer/chattts_plus_trt.yaml`. 
* Webui with Pytorch: `python webui.py --cfg configs/infer/chattts_plus.yaml`

<video src="https://github.com/user-attachments/assets/bd2c1e48-6339-4ad7-bcfa-ed008c992594" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

### License
ChatTTSPlus继承[ChatTTS](https://github.com/2noise/ChatTTS)的license，请以[ChatTTS](https://github.com/2noise/ChatTTS)为标准。

The code is published under AGPLv3+ license.

The model is published under CC BY-NC 4.0 license. It is intended for educational and research use, and should not be used for any commercial or illegal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data used in this repo, are for academic and research purposes only. The data obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.

### 免责声明
我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规.