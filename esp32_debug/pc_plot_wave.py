"""
plot_wave.py — 心音波形 & 频谱调试工具（在电脑上运行）

用法：
  python plot_wave.py debug_raw_xxx.wav debug_2k_xxx.wav   # 同时对比两条
  python plot_wave.py debug_raw_xxx.wav                    # 只看原始
  python plot_wave.py                                      # 自动找当前目录最新的 debug_*.wav
"""

import sys
import os
import glob
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        fs        = wf.getframerate()
        n_frames  = wf.getnframes()
        raw       = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return fs, audio


def plot_file(ax_wave, ax_spec, audio, fs, title):
    t = np.linspace(0, len(audio) / fs, len(audio))

    # 波形
    ax_wave.plot(t, audio, color='#007acc', linewidth=0.6)
    ax_wave.set_title(title, fontsize=10)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylim(-1.1, 1.1)
    ax_wave.grid(True, linestyle='--', alpha=0.4)

    # 频谱图（只显示 0–500Hz 心音关心的频段）
    ax_spec.specgram(audio, NFFT=512, Fs=fs, noverlap=256, cmap='magma')
    ax_spec.set_ylim(0, 500)
    ax_spec.axhline(y=20,  color='lime',  linewidth=1, linestyle='--', label='20 Hz')
    ax_spec.axhline(y=400, color='yellow', linewidth=1, linestyle='--', label='400 Hz')
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.legend(loc='upper right', fontsize=8)
    ax_spec.set_title(f"Spectrogram (0–500 Hz)  fs={fs}Hz", fontsize=10)


def find_latest_debug_files():
    files = sorted(glob.glob("debug_*.wav"), reverse=True)
    if not files:
        return []
    # 找最新时间戳的那组
    ts = files[0].split('_')[-1].replace('.wav', '')
    return [f for f in files if ts in f]


def main():
    if len(sys.argv) > 1:
        wav_files = sys.argv[1:]
    else:
        wav_files = find_latest_debug_files()
        if not wav_files:
            print("❌ 未找到 debug_*.wav，请指定文件路径")
            print("   用法: python plot_wave.py <file1.wav> [file2.wav]")
            return
        print(f"自动加载: {wav_files}")

    n = len(wav_files)
    fig = plt.figure(figsize=(14, 4 * n))
    gs  = gridspec.GridSpec(n * 2, 1, hspace=0.5)

    for i, filepath in enumerate(wav_files):
        if not os.path.exists(filepath):
            print(f"❌ 找不到文件: {filepath}")
            continue
        fs, audio = load_wav(filepath)
        duration  = len(audio) / fs
        name      = os.path.basename(filepath)
        print(f"✅ {name}  采样率={fs}Hz  时长={duration:.2f}s  样本数={len(audio)}")

        ax_wave = fig.add_subplot(gs[i * 2])
        ax_spec = fig.add_subplot(gs[i * 2 + 1])
        plot_file(ax_wave, ax_spec, audio, fs, f"{name}  ({fs}Hz, {duration:.1f}s)")

    plt.suptitle("Heart Sound Debug — Waveform & Spectrogram", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
