# Emotion-LLaMA on CMU-MOSEI

## Setup

### Preparing the Code and Environment

```shell
conda env create -f environment.yaml
```

### Preparing the Pretrained LLM Weights

1. Download the Llama-2-7b-chat-hf model from Huggingface
2. Specify the path to Llama-2 in the [model config file](minigpt4/configs/models/minigpt_v2.yaml#L14):

```yaml
# Set Llama-2-7b-chat-hf path
llama_model: "/home/user/project/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf"
```

3. Specify the path to Emotion-LLaMA in the [config file](train_configs\mosei_finetune.yaml#L9):

```yaml
# Set Emotion-LLaMA path
ckpt: "/home/user/project/Emotion-LLaMA/checkpoints/Emoation_LLaMA.pth"
```

### Preparing the MOSEI Data

1. Download the MOSEI Data on [BaiduDisk]()
2. Unzip the data
```shell
unzip data.zip
```
3. Specify the path to MOSEI Data in the [MOSEI_dataset.py](minigpt4\datasets\datasets\MOSEI_dataset.py#L61)
4. Specify the ann_path and image_path in the [config file](minigpt4\configs\datasets\mosei\mosei.yaml)

## How to train ? 

```shell
python train.py --cfg-path train_configs\mosei_finetune.yaml
```

## How to evaluate ?

```shell
python eval_emotion.py --cfg-path eval_configs\eval_mosei.yaml --dataset mosei_data_builder
```
