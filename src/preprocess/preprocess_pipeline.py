import os
import sys
import numpy as np

# 确保能找到项目模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size

def preprocess_wav_for_pi(wav_path, config):
    """
    完全复刻训练时的预处理流程，用于真机推理。
    返回: list of numpy arrays, 每个形状为 (1, 1, 32, 64)
    """
    sr = config["data"]["sample_rate"]
    mel_cfg = config["mel"]
    
    # 1. 加载音频
    y, _ = load_wav(wav_path, target_sr=sr)
    
    # 2. 带通滤波
    bp = config["data"]["bandpass"]
    y_filtered = apply_bandpass(y, fs=sr, lowcut=bp["low"], highcut=bp["high"])

    # 3. 切片处理
    segments = segment_audio(y_filtered, sr=sr)

    processed_segments = []

    for seg in segments:
        # 4. 转换为 Log-Mel 频谱
        mel = logmel_fixed_size(
            y=seg,
            sr=sr,
            mel_cfg=mel_cfg,
            target_shape=(mel_cfg["n_mels"], mel_cfg["target_frames"]),
        )
        
        # 5. 调整维度适配 TFLite 输入
        # (32, 64) -> (1, 1, 32, 64)  即 [Batch, Channel, Height, Width]
        mel_tensor = mel[np.newaxis, np.newaxis, ...]
        processed_segments.append(mel_tensor.astype(np.float32))
        
    return processed_segments

# 在 preprocess_pipeline.py 中添加
def preprocess_array_for_pi(audio_array, config):
    """
    专门为蓝牙实时流设计：不再读取文件，直接处理内存中的 numpy 数组。
    支持任意长度输入，内部切片（不足补零），与 preprocess_wav_for_pi 完全对齐。
    返回: list of numpy arrays, 每个形状为 (1, 1, 32, 64)
    """
    sr = config["data"]["sample_rate"]
    mel_cfg = config["mel"]

    # 峰值归一化，对齐 load_wav 的行为（训练时每段峰值=1.0）
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        audio_array = audio_array / max_val

    # 带通滤波
    bp = config["data"]["bandpass"]
    y_filtered = apply_bandpass(audio_array, fs=sr, lowcut=bp["low"], highcut=bp["high"])

    # 切片（不足一片时补零）
    segments = segment_audio(y_filtered, sr=sr)

    processed_segments = []
    for seg in segments:
        mel = logmel_fixed_size(
            y=seg,
            sr=sr,
            mel_cfg=mel_cfg,
            target_shape=(mel_cfg["n_mels"], mel_cfg["target_frames"]),
        )
        processed_segments.append(mel[np.newaxis, np.newaxis, ...].astype(np.float32))

    return processed_segments

if __name__ == "__main__":
    # 快速自测逻辑
    import yaml
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        conf = yaml.safe_load(f)
        
    test_wav = "/home/rasp4b/FypPi/data/raw/Dataset2/training-a/a0001.wav" # 找一个本地 wav 测试
    if os.path.exists(test_wav):
        tensors = preprocess_wav_for_pi(test_wav, conf)
        print(f"✅ 预处理完成，生成了 {len(tensors)} 个切片")
        print(f"✅ 单个切片形状: {tensors[0].shape}")