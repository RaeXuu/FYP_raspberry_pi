import yaml
from pathlib import Path
import numpy as np

# =========================
# Load config
# =========================
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]

# =========================
# Segment function
# =========================
def segment_audio(y, sr):
    """
    根据 config.yaml 中的参数切片音频
    """
    segment_sec = data_cfg["segment_length"]
    overlap = data_cfg["overlap"]

    seg_len = int(sr * segment_sec)
    hop = int(seg_len * (1 - overlap))

    if len(y) == 0:
        return []

    segments = []

    for start in range(0, len(y) - seg_len + 1, hop):
        end = start + seg_len
        seg = y[start:end]

        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)), mode="constant")

        segments.append(seg)

    return segments


# =========================
# Self test
# =========================
if __name__ == "__main__":
    import pandas as pd
    from src.preprocess.load_wav import load_wav
    from src.preprocess.filters import apply_bandpass

    df = pd.read_csv("data/metadata_physionet.csv")
    path = df.iloc[0]["filepath"]

    print("测试样本:", path)

    # Step1: load wav
    y, sr = load_wav(path, target_sr=data_cfg["sample_rate"])

    # Step2: bandpass
    y = apply_bandpass(
        y,
        fs=sr,
        lowcut=data_cfg["bandpass"]["low"],
        highcut=data_cfg["bandpass"]["high"]
    )

    # Step3: segment
    segments = segment_audio(y, sr)

    print("切片数量:", len(segments))
    print("单个片段长度:", len(segments[0]))
    print("理论长度:", sr * data_cfg["segment_length"])
    print("切片测试完成 ✅")
