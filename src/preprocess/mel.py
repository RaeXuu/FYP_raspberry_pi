
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
    import yaml
    import pandas as pd
    from pathlib import Path
    from src.preprocess.load_wav import load_wav
    from src.preprocess.filters import apply_bandpass
    from src.preprocess.segment import segment_audio

    CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    mel_cfg = cfg["mel"]
    sr = cfg["data"]["sample_rate"]
    bp = cfg["data"]["bandpass"]

    df = pd.read_csv("data/metadata_physionet.csv")
    path = df.iloc[0]["filepath"]
    print("测试样本:", path)

    y, _ = load_wav(path, target_sr=sr)
    y = apply_bandpass(y, fs=sr, lowcut=bp["low"], highcut=bp["high"])
    segments = segment_audio(y, sr=sr)
    seg = segments[0]

    print("单段长度:", len(seg))

    mel = logmel_fixed_size(seg, sr=sr, mel_cfg=mel_cfg,
                            target_shape=(mel_cfg["n_mels"], mel_cfg["target_frames"]))
    print("Log-Mel shape:", mel.shape)
    print("Mel 测试完成 ✅")