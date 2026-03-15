# 做一个 Butterworth 带通滤波器
# 典型心音频段：20–400 Hz（参数可以改）

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]


import numpy as np
from scipy.signal import butter, filtfilt

def design_bandpass(lowcut, highcut, fs, order=5):
    """
    设计 Butterworth 带通滤波器
    lowcut, highcut: 截止频率（Hz）
    fs: 采样率（Hz）
    order: 滤波器阶数
    """
    nyq = 0.5 * fs  # Nyquist 频率
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(y, fs, lowcut=None, highcut=None, order=5):
    """
    对一条音频信号做带通滤波。

    参数:
        y (np.ndarray): 输入波形
        fs (int): 采样率
        lowcut (float): 低截止频率
        highcut (float): 高截止频率
        order (int): 滤波器阶数

    返回:
        y_filt (np.ndarray): 滤波后的波形
    """
    if y is None or len(y) == 0:
        return y
    
    if lowcut is None:
        lowcut = data_cfg["bandpass"]["low"]
    if highcut is None:
        highcut = data_cfg["bandpass"]["high"]

    b, a = design_bandpass(lowcut, highcut, fs, order=order)
    # filtfilt 双向滤波，零相位失真
    y_filt = filtfilt(b, a, y)
    return y_filt


if __name__ == "__main__":
    """
    简单自测：
    1. 从 metadata 里取第一条
    2. 用 load_wav 读出来
    3. 做带通滤波，并打印长度、能量对比
    """
    import pandas as pd
    from src.preprocess.load_wav import load_wav

    meta_path = "/home/rasp4b/FypPi/data/metadata_physionet.csv"
    df = pd.read_csv(meta_path)

    first_path = df.iloc[0]["filepath"]
    print("测试样本路径:", first_path)

    y, sr = load_wav(first_path, target_sr=2000)
    print("原始信号长度:", len(y))

    y_filt = apply_bandpass(y, fs=sr, lowcut=25, highcut=400, order=5)
    print("滤波后信号长度:", len(y_filt))

    # 简单对比能量
    orig_energy = np.mean(y**2)
    filt_energy = np.mean(y_filt**2)
    print("原始能量:", orig_energy)
    print("滤波后能量:", filt_energy)
    print("带通滤波测试完成 ✅")
