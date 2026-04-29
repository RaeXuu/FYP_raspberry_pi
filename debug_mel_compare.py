"""
对比 Pi/iOS Mel 预处理，打印每一步中间值。
对齐 iOS 端 DebugPreprocess.swift 的输出格式。
"""
import os, sys
import numpy as np
import yaml
import librosa

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass

from scipy.signal import butter                                                                                                                                
sos = butter(5, [25/1000, 400/1000], btype='band', output='sos')                                                                                               
print("[Debug] SOS:")                                                                                                                                          
for row in sos:                                                                                                                                                
    print(f"  {[f'{v:.15f}' for v in row]}")  

# ── 配置 ──
TEST_WAV = "/home/rasp4b/FypPi/data/raw/DataSet2/training-a/a0002.wav"

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

sr   = config["data"]["sample_rate"]
mel  = config["mel"]
bp   = config["data"]["bandpass"]

print("=" * 60)
print(f"🎬 {os.path.basename(TEST_WAV)}")
print("=" * 60)

# ── Step 1: 加载 ──
y, _ = load_wav(TEST_WAV, target_sr=sr)
y = y[:40000]  # 取前20s
if len(y) < 40000:
    y = np.pad(y, (0, 40000 - len(y)))
print(f"\n[Debug] 原始: first10={[f'{v:.6f}' for v in y[:10]]}")

# ── Step 2: 带通滤波 ──
filt = apply_bandpass(y, fs=sr, lowcut=bp["low"], highcut=bp["high"])
print(f"[Debug] 滤波后: first10={[f'{v:.6f}' for v in filt[:10]]}")
print(f"[Debug] 滤波后[100..<110]: {[f'{v:.6f}' for v in filt[100:110]]}")    

# ── Step 3: 第一个窗口 + 峰值归一化 ──
window = filt[:4000].copy()
max_val = np.max(np.abs(window))
if max_val > 0:
    window = window / max_val
print(f"[Debug] 归一化后: first10={[f'{v:.6f}' for v in window[:10]]} max={max_val:.6f}")

# ── Step 4: STFT 第一帧 (对齐 librosa.melspectrogram internals) ──
S = librosa.stft(
    window,
    n_fft=mel["n_fft"],
    hop_length=mel["hop_length"],
    win_length=mel.get("win_length", mel["n_fft"]),
    center=True,
    window="hann",
)
power = np.abs(S) ** mel.get("power", 2.0)
stft0 = power[:, 0]
print(f"[Debug] STFT[0]: first10={[f'{v:.2f}' for v in stft0[:10]]}")

# ── Step 5: Mel spectrogram (librosa) ──
mel_spec = librosa.feature.melspectrogram(
    y=window, sr=sr,
    n_fft=mel["n_fft"], hop_length=mel["hop_length"],
    win_length=mel.get("win_length", mel["n_fft"]),
    n_mels=mel["n_mels"], fmin=mel.get("fmin", 0),
    fmax=mel.get("fmax", None), power=mel.get("power", 2.0),
)
logmel = librosa.power_to_db(mel_spec + 1e-6)
logmel = librosa.util.fix_length(logmel, size=mel["target_frames"], axis=1)
flat = logmel.flatten()

print(f"[Debug] Mel[0..<10] = {[f'{v:.6f}' for v in flat[:10]]}")
print(f"[Debug] Mel[64..<74] = {[f'{v:.6f}' for v in flat[64:74]]}")
print(f"[Debug] Mel min={flat.min():.6f} max={flat.max():.6f} mean={flat.mean():.6f}")
print("=" * 60)
                                 
                                               