## ChatTTSPlus Voice Cloning
Currently supports two methods of voice cloning:
* Training with lora 
* Training speaker embedding

Note:
* The voice cloning feature of ChatTTSPlus is for learning purposes only. Please do not use it for illegal or criminal activities. We take no responsibility for any illegal use of the codebase.
* Some code references: [ChatTTS PR](https://github.com/2noise/ChatTTS/pull/680)

### Data Collection and Preprocessing
* Prepare over 30 minutes of audio from the person you want to clone.
* Process using GPT-SoViTs preprocessing workflow, executing in sequence: audio splitting, UVR5 background separation, noise reduction, speech-to-text, etc.
* Finally get a `.list` file in this format: speaker_name | audio_path | lang | text, like this:
```text 
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_1.WAV_10.wav_0000000000_0000152640.wav|EN|Hehe, I watched Parasite recently, really recommend it.
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_1.WAV_10.wav_0000152640_0000323520.wav|EN|The plot is tight and the filming technique is very unique.
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_1.WAV_10.wav_0000323520_0000474880.wav|EN|It won many awards too, what type of movies do you like?
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_2.WAV_10.wav_0000000000_0000114560.wav|EN|I like mystery films, any other recommendations?
xionger|E:\my_projects\ChatTTSPlus\data\xionger\slicer_opt\vocal_3.WAV_10.wav_0000000000_0000133760.wav|EN|Woof woof woof, then you must watch And Then There Were None.
```

### Model Training
#### Lora Training (Recommended)
* Modify `DATA/meta_infos` in `configs/train/train_voice_clone_lora.yaml` to the `.list` file processed in previous step
* Modify `exp_name` in `configs/train/train_voice_clone_lora.yaml` to the experiment name, preferably using speaker_name for identification.
* Then run `accelerate launch train_lora.py --config configs/train/train_voice_clone_lora.yaml` to start training.
* Trained models will be saved in the `outputs` folder, like `outputs/xionger_lora-1732809910.2932503/checkpoints/step-900`
* You can visualize training logs using tensorboard, e.g., `tensorboad --logdir=outputs/xionger_lora-1732809910.2932503/tf_logs`

#### Speaker Embedding Training (Not Recommended, Hard to Converge)
* Modify `DATA/meta_infos` in `configs/train/train_speaker_embedding.yaml` to the `.list` file processed in previous step
* Modify `exp_name` in `configs/train/train_speaker_embedding.yaml` to the experiment name, preferably using speaker_name for identification.
* Then run `accelerate launch train_lora.py --config configs/train/train_speaker_embedding.yaml` to start training.
* Trained speaker embeddings will be saved in the `outputs` folder, like `outputs/xionger_speaker_emb-1732931630.7137222/checkpoints/step-1/xionger.pt`
* You can visualize training logs using tensorboard, e.g., `tensorboad --logdir=outputs/xionger_speaker_emb-1732931630.7137222/tf_logs`

### Model Inference
* Launch webui: `python webui.py --cfg configs/infer/chattts_plus.yaml`
* Refer to the following video tutorial for usage:

<video src="https://github.com/user-attachments/assets/bd2c1e48-6339-4ad7-bcfa-ed008c992594" controls="controls" width="500" height="300">Your browser doesn't support playing this video!</video>