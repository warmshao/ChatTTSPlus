## ChatTTSPlus 声音克隆
目前支持两种声音克隆的方式:
* 训练lora
* 训练speaker embedding

注意:
* ChatTTSPlus 的声音克隆功能仅供学习使用，请勿用于非法或犯罪活动。我们对代码库的任何非法使用不承担责任。
* 部分代码参考: [ChatTTS PR](https://github.com/2noise/ChatTTS/pull/680)


### 数据收集和预处理
* 准备想要克隆的某个人的30分钟以上的音频。
* 使用GPT-SoViTs的预处理流程处理，依次执行：音频切分、UVR5背景声分离、语音降噪、语音识别文本等。
* 最后得到这样的`.list`文件: speaker_nam | audio_path | lang | text，类似这样:
```text
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_1.WAV_10.wav_0000000000_0000152640.wav|ZH|嘿嘿，最近我看了寄生虫，真的很推荐哦。
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_1.WAV_10.wav_0000152640_0000323520.wav|ZH|这部电影剧情紧凑，拍摄手法也很独特。
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_1.WAV_10.wav_0000323520_0000474880.wav|ZH|还得了很多奖项，你有喜欢的电影类型吗？
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_2.WAV_10.wav_0000000000_0000114560.wav|ZH|我喜欢悬疑片，有其他推荐吗？
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_3.WAV_10.wav_0000000000_0000133760.wav|ZH|汪汪汪，那你一定要看无人生还。
```

### 模型训练:
#### lora训练(推荐):
* 修改`configs/train/train_voice_clone_lora.yaml`里 `DATA/meta_infos`为上一个处理到的`.list`文件
* 修改`configs/train/train_voice_clone_lora.yaml`里 `exp_name`为实验的名称，最好用speaker_name做区分识别。
* 然后运行`accelerate launch train_lora.py --config configs/train/train_voice_clone_lora.yaml`开始训练。
* 训练的模型会保存在: `outputs` 文件夹下，比如`outputs/xionger_lora-1732809910.2932503/checkpoints/step-900`
* 你可以使用tensorboard可视化训练的log, 比如`tensorboad --logdir=outputs/xionger_lora-1732809910.2932503/tf_logs`

#### speaker embedding训练(不推荐，很难收敛):
* 修改`configs/train/train_speaker_embedding.yaml`里 `DATA/meta_infos`为上一个处理到的`.list`文件
* 修改`configs/train/train_speaker_embedding.yaml`里 `exp_name`为实验的名称，最好用speaker_name做区分识别。
* 然后运行`accelerate launch train_lora.py --config configs/train/train_speaker_embedding.yaml`开始训练。
* 训练的speaker embedding 会保存在: `outputs` 文件夹下，比如`outputs/xionger_speaker_emb-1732931630.7137222/checkpoints/step-1/xionger.pt`
* 你可以使用tensorboard可视化训练的log, 比如`tensorboad --logdir=outputs/xionger_speaker_emb-1732931630.7137222/tf_logs`

### 模型推理
* 启动webui: ` python webui.py --cfg configs/infer/chattts_plus.yaml`
* 参考以下视频教程使用:

<video src="https://github.com/user-attachments/assets/bd2c1e48-6339-4ad7-bcfa-ed008c992594" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>
