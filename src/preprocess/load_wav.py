import yaml
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import os

# 保持你原来的配置读取逻辑
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]

def load_wav(filepath, target_sr=None):
    """
    加载 WAV 文件并做基础预处理：
    1. 读取音频
    2. 重采样到 target_sr
    3. 幅度归一化到 [-1, 1]
    """
    if target_sr is None:
        target_sr = data_cfg["sample_rate"]

    # 读取原始音频
    y, sr = librosa.load(filepath, sr=None)

    # 重采样
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 幅度归一化
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr


def batch_load_from_metadata(df, sr=None):
    """
    根据 metadata DataFrame 批量加载音频。
    返回 list，每个元素包含 {audio, sr, label, filepath}
    """
    audio_items = []

    for idx, row in df.iterrows():
        filepath = row["filepath"]
        
        # 简单检查路径是否存在
        if not os.path.exists(filepath):
            continue
            
        y, s = load_wav(filepath, target_sr=sr)

        audio_items.append({
            "audio": y,
            "sr": s,
            "label": row.get("label"),
            "fname": row.get("fname"),           # 对应新 CSV 的文件名
            "dataset": row.get("source_folder"), # 将 source_folder 映射到你的 dataset 键上
            "filepath": filepath
        })

    return audio_items


if __name__ == "__main__":
    # 1. 这里的路径改为我们刚生成的 PhysioNet 元数据
    df = pd.read_csv("data/metadata_physionet.csv")
    
    # 2. 这里的 sr 保持你之前的逻辑，或者改为 data_cfg["sample_rate"]
    target_sr = data_cfg["sample_rate"]
    
    # 先加载前一个测试
    audios = batch_load_from_metadata(df.head(1), sr=target_sr)  

    print("测试加载成功，返回数量:", len(audios))
    if len(audios) > 0:
        print("第一个样本信息:")
        print("  文件名:", audios[0]["fname"])
        print("  采样率:", audios[0]["sr"])
        print("  音频长度:", len(audios[0]["audio"]))
        print("  标签:", audios[0]["label"])