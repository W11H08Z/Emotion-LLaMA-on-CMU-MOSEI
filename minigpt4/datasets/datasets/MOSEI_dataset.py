import os
import cv2
import wave
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset

from transformers import AutoImageProcessor, VideoMAEModel
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from transformers import ViTImageProcessor, ViTMAEModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)

device = "cuda"
hubert_path = "/home/peiyuan/zhw/model/chinese-hubert-large"
videomae_path = "/home/peiyuan/zhw/model/videomae-large"
vitmae_path = "/home/peiyuan/zhw/model/vit-mae-large"

class MOSEI_Dataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        ann_path: label_path
        vis_root: data_root
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.emotion_instruction_pool = "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt."
        self.task_pool = "emotion"
        

        self.emos = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(self.emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(self.emos): self.idx2emo[ii] = emo

        self.data_path = vis_root
        self.labels_path = ann_path

        self.audio_path = os.path.join(self.data_path, 'Raw', 'Audio', 'Full', 'WAV_16000')
        self.text_path = os.path.join(self.data_path, 'Raw', 'Transcript', 'Segmented', 'Combined')
        self.vision_path = os.path.join(self.data_path, 'Raw', 'Videos', 'Full', 'Combined')

        mode = "test" # train, val, test
        self.labels_df = pd.read_csv(self.labels_path)
        self.labels_df = self.labels_df[self.labels_df['mode'] == mode]

        self.load_batch_size = 16
        self.save_path = f"/home/peiyuan/zhw/MOSEI-Emotion-LLaMA/data/MOSEI/{mode}"
        self.hubert_feat_path = os.path.join(self.save_path, 'hubert_feat')
        self.vitmae_feat_path = os.path.join(self.save_path, 'vitmae_feat')
        self.videomae_feat_path = os.path.join(self.save_path, 'videomae_feat')
        self.image_path = os.path.join(self.save_path, 'image')

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            os.mkdir(self.hubert_feat_path)
            os.mkdir(self.vitmae_feat_path)
            os.mkdir(self.videomae_feat_path)
            os.mkdir(self.image_path)
            self.extract_feature()

    def extract_feature(self):
        hubert_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
        hubert_model = HubertModel.from_pretrained(hubert_path)
        videomae_processor = AutoImageProcessor.from_pretrained(videomae_path)
        videomae_model = VideoMAEModel.from_pretrained(videomae_path)
        vitmae_processor = AutoImageProcessor.from_pretrained(vitmae_path)
        vitmae_model = ViTMAEModel.from_pretrained(vitmae_path)

        id_df = self.labels_df['video_id']
        clip_df = self.labels_df['clip_id']

        for i in tqdm(range(0, len(id_df), self.load_batch_size), total=len(id_df) // self.load_batch_size,
                      unit='batch'):
            vid_list = id_df.iloc[i:i + self.load_batch_size].to_list()
            clip_list = clip_df.iloc[i:i + self.load_batch_size].to_list()
            vid_list, clip_list, video_frames, audio_batch, frame_rate = self.get_vision_audio_data(vid_list, clip_list)

            # chinese hubert
            hubert_input_values = hubert_extractor(audio_batch, sampling_rate=frame_rate, return_tensors="pt", padding=True, \
                                              max_length= 10 * frame_rate, \
                                              truncation=True).input_values.float().to(device)
            # print("input_values:", input_values)
            hubert_model.eval().to(device)
            with torch.no_grad():
                hubert_outputs = hubert_model(hubert_input_values, output_hidden_states=True) # tuple of (B, T, D)
                hubert_feature = hubert_outputs.last_hidden_state.detach()  # sum, (B, T, D)
                print(hubert_feature.shape)
                hubert_feature = torch.mean(hubert_feature, dim=1, keepdim=True)

            # vitmae
            vitmae_input_frames = np.array([frames[0] for frames in video_frames])
            vitmae_inputs = vitmae_processor(images=vitmae_input_frames, return_tensors="pt").to(device)
            vitmae_model.eval().to(device)
            with torch.no_grad():
                vitmae_outputs = vitmae_model(**vitmae_inputs)
                vitmae_feature = vitmae_outputs.last_hidden_state.detach()
                print(vitmae_feature.shape)
                vitmae_feature = torch.mean(vitmae_feature, dim=1, keepdim=True)
  
            # videomae
            video_frames = np.array(video_frames)
            seq_len, img_shape = video_frames[0].shape[0], (video_frames[0].shape[1], video_frames[0].shape[2], video_frames[0].shape[3])
            videomae_pixel_values = videomae_processor(list(video_frames.reshape(-1, *img_shape)), return_tensors="pt").pixel_values.reshape(-1, seq_len, 3, 224, 224).to(device)
            videomae_model.eval().to(device)
            with torch.no_grad():
                videomae_outputs = videomae_model(videomae_pixel_values)
                videomae_feature = videomae_outputs.last_hidden_state.detach()
                print(videomae_feature.shape)
                videomae_feature = torch.mean(videomae_feature, dim=1, keepdim=True)

            for i, vid in enumerate(vid_list):
                np.save(os.path.join(self.hubert_feat_path, f"{vid}_{clip_list[i]}.npy"), hubert_feature[i].cpu().numpy())
                np.save(os.path.join(self.vitmae_feat_path, f"{vid}_{clip_list[i]}.npy"), vitmae_feature[i].cpu().numpy())
                np.save(os.path.join(self.videomae_feat_path, f"{vid}_{clip_list[i]}.npy"), videomae_feature[i].cpu().numpy())
        
        for index in tqdm(range(0, len(self.labels_df))):
            t = self.labels_df.iloc[index]
            vid, clip = t['video_id'], t['clip_id']
            frame = extract_frame(vid, clip, self.text_path, self.vision_path)
            np.save(os.path.join(self.image_path, f"{vid}_{clip}.npy"), frame)


    def get_vision_audio_data(self, video_id_list, clip_id_list):
        frame_rate = None
        audio_batch = []
        video_frames = []

        def task(vid_clip):
            vid, clip = vid_clip
            return process_video_audio(vid, clip, self.text_path, self.vision_path, self.audio_path)

        with ThreadPoolExecutor(max_workers=80) as executor:
            results = list(executor.map(task, zip(video_id_list, clip_id_list)))

        vid_list = []
        clip_list = []
        for vid, clip, face_frames, audio_array, cur_frame_rate, _ in results:
            vid_list.append(vid)
            clip_list.append(clip)
            video_frames.append(face_frames)
            audio_batch.append(audio_array)
            if frame_rate is None:
                frame_rate = cur_frame_rate

        return vid_list,  clip_list, video_frames, audio_batch, frame_rate

    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, index):
        t = self.labels_df.iloc[index]
        vid, clip = t['video_id'], t['clip_id']

        frame = np.load(os.path.join(self.image_path, f"{vid}_{clip}.npy"))
        image = Image.fromarray(frame.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)

        hubert_feat, vitmae_feat, videomae_feat = self.get(vid, clip)
        video_features = torch.cat((vitmae_feat, videomae_feat, hubert_feat), dim=0)

        caption = self.emos[t[self.emos].to_list().index(max(t[self.emos].to_list()))]
        caption = self.text_processor(caption)
        instruction_pool = self.emotion_instruction_pool

        emotion = self.emo2idx[caption]
        sentence = t['text']
        character_line = "The person in video says: {}. ".format(sentence)

        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, self.task_pool, instruction_pool)


        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": f"{vid}_{clip}"
        }
        
    
    def get(self, vid, clip):
        hubert_feat = torch.tensor(np.load(os.path.join(self.hubert_feat_path, f"{vid}_{clip}.npy")))
        vitmae_feat = torch.tensor(np.load(os.path.join(self.vitmae_feat_path, f"{vid}_{clip}.npy")))
        videomae_feat = torch.tensor(np.load(os.path.join(self.videomae_feat_path, f"{vid}_{clip}.npy")))

        return hubert_feat, vitmae_feat, videomae_feat


def extract_frame(vid, clip, text_path, vision_path):
    with open(os.path.join(text_path, vid + '.txt'), 'r', encoding='utf-8') as file:
        for line in file:
            info = line.split('___')
            assert len(info) >= 5
            cur_id, cur_clip, start, end = info[:4]
            if int(cur_clip) == int(clip):
                video_path = os.path.join(vision_path, vid + '.mp4')
                cap = cv2.VideoCapture(video_path)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                start_frame = max(0, int(float(start) * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()

                return frame_rgb

def process_video_audio(vid, clip, text_path, vision_path, audio_path):
    audio_array = None
    frame_rate = None

    with open(os.path.join(text_path, vid + '.txt'), 'r', encoding='utf-8') as file:
        for line in file:
            info = line.split('___')
            assert len(info) >= 5
            cur_id, cur_clip, start, end = info[:4]
            if int(cur_clip) == int(clip):
                frames = load_video(os.path.join(vision_path, vid + '.mp4'), 1, float(start), float(end))
                video = sample_frame(frames, clip_len=16)
                face_frames = extract_faces_from_frames(video)
                if len(face_frames) > 0:
                    if len(face_frames) < 16:
                        last_frame = face_frames[-1]
                        face_frames.extend([last_frame] * (16 - len(face_frames)))
                    face_frames = face_frames[:16]
                else:
                    face_frames = [np.zeros((128, 128, 3))]*16


                audio_array, cur_frame_rate = load_audio(os.path.join(audio_path, vid + '.wav'), float(start), float(end))
                frame_rate = cur_frame_rate

                return vid, clip, face_frames, audio_array, frame_rate, frames

def load_audio(audio_path, start_time, end_time):
    with wave.open(audio_path, 'rb') as wav_file:
        num_channels, sample_width, frame_rate, num_frames, comptype, compname = wav_file.getparams()

        # 将时间转换为帧数
        start_frame = int(start_time * frame_rate)  # a是开始时间（秒）
        end_frame = int(end_time * frame_rate)  # b是结束时间（秒）

        # 确保开始和结束帧在文件帧数范围内
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        # 读取指定范围的帧
        wav_file.setpos(start_frame)
        frames = wav_file.readframes(end_frame - start_frame)

    # 将帧数据转换为NumPy数组
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError("Unsupported sample width")

    audio_array = np.frombuffer(frames, dtype=dtype)
    return audio_array, frame_rate


def load_video(file_path, interval, start_time, end_time):
    # 打开视频文件
    cap = cv2.VideoCapture(file_path)
    frames = []

    # cv2.CAP_PROP_FRAME_COUNT 是 OpenCV 中的一个属性标识符，用于获取视频文件中的总帧数。
    # 当你使用 cap.get(cv2.CAP_PROP_FRAME_COUNT) 时，它会返回视频文件中的帧数，这个值是一个整数。
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = max(0, int(start_time * fps))
    end_frame = min(total_frames, int(end_time * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    num_frames = int((end_frame - start_frame) / interval)
    assert num_frames > 0, "Error: Video file has no frames"
    for i in range(num_frames):
        # cv2.CAP_PROP_POS_FRAMES 代表当前帧位置
        # cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval) 将 cv2.CAP_PROP_POS_FRAMES 设置为 i * interval 帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()  # 从 cv2.CAP_PROP_POS_FRAMES 读取帧，并且当前帧+1
        # ret: Bool 当前帧是否读取成功; frame: array 读取到的帧
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
            frames.append(frame)

    cap.release()
    return frames

def extract_faces_from_frames(frames, target_size=(128, 128)):
    # 加载预训练的Haar特征分类器模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 存储提取并放缩后的人脸
    resized_faces = []

    # 遍历图像目录中的所有文件
    for frame in frames:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 如果检测到多张人脸，只取第一张
        if len(detected_faces) > 0:
            x, y, w, h = detected_faces[0]
            face_img = frame[y:y+h, x:x+w]

            # 放缩人脸到目标尺寸
            resized_face = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
            resized_faces.append(resized_face)

    return resized_faces

def sample_frame(frames, clip_len):
    indices = np.linspace(0, len(frames)-1, num=clip_len).astype(np.int64)
    return np.stack([frames[i] for i in indices])

if __name__ == '__main__':
    MOSEI_Dataset(None, None, "/home/peiyuan/dataset/CMU-MOSEI/data", "/home/peiyuan/dataset/CMU-MOSEI/data/final_multi_labels.csv")