
import numpy as np
import librosa

def wav_to_logmel(
    y,
    sr,
    mel_cfg,
    eps=1e-6
):
    """
    将单段音频转成 Log-Mel Spectrogram。
    输入: y (长度固定的片段，例如 8000 点)
    输出: 2D Mel 特征 (n_mels × time_frames)
    """

    # 1. Mel 滤波器
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg.get("win_length", mel_cfg["n_fft"]),
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax", None),
        power=mel_cfg.get("power", 2.0),
    )
    # 2. 转 Log-Mel
    logmel = librosa.power_to_db(mel + eps)

    return logmel


def logmel_fixed_size(
    y,
    sr,
    mel_cfg,
    target_shape
):
    mel = wav_to_logmel(
        y=y,
        sr=sr,
        mel_cfg=mel_cfg
    )

    mel_resized = librosa.util.fix_length(
        mel,
        size=target_shape[1],
        axis=1
    )

    return mel_resized



if __name__ == "__main__":
    import pandas as pd
    from src.preprocess.load_wav import load_wav
    from src.preprocess.filters import apply_bandpass
    from src.preprocess.segment import segment_audio

    # 1. 定义 mel 专用配置字典 (重点修复这里！)
    # 这里的参数直接决定了生成的频谱图长什么样
    mel_settings = {
        "n_fft": 256,         # 窗长
        "hop_length": 64,     # 帧移 (决定了频谱图的宽度)
        "n_mels": 32,         # Mel 滤波器数量 (决定了频谱图的高度)
        "fmin": 20,           # 最低频率 (心音低频多，20Hz 比较合适)
        "fmax": 400           # 最高频率 (配合你之前的带通滤波)
    }

    df = pd.read_csv("data/metadata_physionet.csv")
    path = df.iloc[0]["filepath"]

    print("测试样本:", path)

    # Step1: load wav
    y, sr = load_wav(path, target_sr=2000)

    # Step2: bandpass
    y = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)

    # Step3: segment
    segments = segment_audio(y, sr=sr)
    seg = segments[0]

    print("单段长度:", len(seg))

    # Step4: log-mel (这里传入刚才定义的字典)
    # 注意：target_shape=(32, 64) 里的 32 最好和上面的 n_mels 一致
    mel = logmel_fixed_size(seg, sr=sr, mel_cfg=mel_settings, target_shape=(32, 64))

    print("Log-Mel shape:", mel.shape)
    print("Mel 测试完成 ✅")