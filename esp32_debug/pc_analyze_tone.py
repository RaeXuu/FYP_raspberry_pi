"""
analyze_tone.py — 正弦波传输质量分析工具（在电脑上运行）

用法：
  python analyze_tone.py 100hz_og.wav 000_2k.wav 100
  python analyze_tone.py 100hz_og.wav 000_raw.wav 100

参数：
  <og_wav>       原始生成的正弦波文件
  <received_wav> 树莓派接收并保存的文件
  <frequency>    目标频率（Hz）
"""

import sys
import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ==========================================
# 加载 WAV
# ==========================================
def load_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        fs       = wf.getframerate()
        n_frames = wf.getnframes()
        raw      = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return fs, audio


# ==========================================
# 核心分析
# ==========================================
def analyze(audio, fs, target_freq):
    N   = len(audio)
    fft = np.fft.rfft(audio * np.hanning(N))
    mag = np.abs(fft) / (N / 2)
    freqs = np.fft.rfftfreq(N, 1 / fs)

    # 目标频率附近的 bin（±2Hz 容差）
    tolerance = max(2.0, fs / N * 2)
    signal_mask = np.abs(freqs - target_freq) <= tolerance

    # 谐波 bin（2x, 3x, 4x 目标频率，各 ±2Hz）
    harmonic_freqs = [target_freq * k for k in range(2, 5)]
    harmonic_mask  = np.zeros(len(freqs), dtype=bool)
    for hf in harmonic_freqs:
        if hf < fs / 2:
            harmonic_mask |= np.abs(freqs - hf) <= tolerance

    noise_mask = ~signal_mask & ~harmonic_mask

    signal_power   = np.sum(mag[signal_mask]   ** 2)
    harmonic_power = np.sum(mag[harmonic_mask] ** 2)
    noise_power    = np.sum(mag[noise_mask]    ** 2)
    total_power    = np.sum(mag ** 2)

    peak_idx   = np.argmax(mag[signal_mask]) if signal_mask.any() else -1
    peak_amp   = mag[signal_mask].max() if signal_mask.any() else 0.0
    peak_freq  = freqs[signal_mask][peak_idx] if signal_mask.any() else 0.0

    snr        = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    purity     = signal_power / total_power * 100 if total_power > 0 else 0.0
    thd        = np.sqrt(harmonic_power / signal_power) * 100 if signal_power > 0 else float('inf')

    return {
        "freqs":          freqs,
        "mag":            mag,
        "peak_freq":      peak_freq,
        "peak_amp":       peak_amp,
        "snr_db":         snr,
        "purity_pct":     purity,
        "thd_pct":        thd,
        "signal_power":   signal_power,
        "harmonic_freqs": harmonic_freqs,
    }


# ==========================================
# 绘图
# ==========================================
def plot_comparison(og, og_res, rx, rx_res, target_freq):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)

    fs_og = og["fs"]
    fs_rx = rx["fs"]

    # ---- 波形 ----
    for col, (audio, fs, res, label, color) in enumerate([
        (og["audio"], fs_og, og_res, f"OG  ({fs_og}Hz)", "#2196F3"),
        (rx["audio"], fs_rx, rx_res, f"Received  ({fs_rx}Hz)", "#F44336"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        t  = np.linspace(0, len(audio) / fs, len(audio))
        ax.plot(t, audio, color=color, linewidth=0.5)
        ax.set_title(f"Waveform — {label}", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.4)

    # ---- 频谱对比（叠在一张图）----
    ax_spec = fig.add_subplot(gs[1, :])
    view_max = min(target_freq * 4, max(fs_og, fs_rx) / 2)

    og_mask = og_res["freqs"] <= view_max
    rx_mask = rx_res["freqs"] <= view_max

    og_db = 20 * np.log10(og_res["mag"][og_mask] + 1e-9)
    rx_db = 20 * np.log10(rx_res["mag"][rx_mask] + 1e-9)

    ax_spec.plot(og_res["freqs"][og_mask], og_db, color="#2196F3", linewidth=1,   label="OG",       alpha=0.8)
    ax_spec.plot(rx_res["freqs"][rx_mask], rx_db, color="#F44336", linewidth=1,   label="Received", alpha=0.8)
    ax_spec.axvline(x=target_freq, color="gold", linewidth=1.2, linestyle='--', label=f"{target_freq}Hz (target)")
    for hf in rx_res["harmonic_freqs"]:
        if hf <= view_max:
            ax_spec.axvline(x=hf, color="orange", linewidth=0.8, linestyle=':', alpha=0.7)

    ax_spec.set_title("Frequency Spectrum Comparison (dB)", fontsize=10)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude (dB)")
    ax_spec.legend(fontsize=9)
    ax_spec.grid(True, linestyle='--', alpha=0.4)

    # ---- 指标汇总 ----
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    attenuation = (20 * np.log10(rx_res["peak_amp"] / og_res["peak_amp"])
                   if og_res["peak_amp"] > 0 else float('-inf'))

    rows = [
        ["Metric",         "OG Signal",                         "Received Signal",                   "Description"],
        ["Peak Freq",      f"{og_res['peak_freq']:.2f} Hz",     f"{rx_res['peak_freq']:.2f} Hz",     f"Target: {target_freq} Hz"],
        ["SNR",            f"{og_res['snr_db']:.1f} dB",        f"{rx_res['snr_db']:.1f} dB",        ">20dB acceptable, >40dB good"],
        ["Purity",         f"{og_res['purity_pct']:.2f}%",      f"{rx_res['purity_pct']:.2f}%",      "Target freq energy / total energy"],
        ["THD",            f"{og_res['thd_pct']:.2f}%",         f"{rx_res['thd_pct']:.2f}%",         "<5% good, >10% distorted"],
        ["Attenuation",    "—",                                 f"{attenuation:.1f} dB",             "Negative = loss, ~0 = faithful"],
    ]

    col_widths = [0.18, 0.22, 0.22, 0.38]
    x_starts   = [0.01, 0.20, 0.43, 0.66]
    y_start    = 0.95
    row_h      = 0.13

    for r, row in enumerate(rows):
        y = y_start - r * row_h
        bg = "#1a1a2e" if r == 0 else ("#16213e" if r % 2 == 0 else "#0f3460")
        ax_table.add_patch(plt.Rectangle((0, y - row_h * 0.85), 1.0, row_h * 0.9,
                                         color=bg, transform=ax_table.transAxes,
                                         clip_on=False, zorder=0))
        for c, (text, x) in enumerate(zip(row, x_starts)):
            weight = 'bold' if r == 0 else 'normal'
            color  = 'gold' if r == 0 else 'white'
            ax_table.text(x, y - row_h * 0.3, text, transform=ax_table.transAxes,
                          fontsize=8.5, color=color, fontweight=weight, va='center')

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.set_title("Quantitative Metrics", fontsize=10, pad=8)

    plt.suptitle(f"Tone Analysis — {target_freq} Hz", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ==========================================
# 主程序
# ==========================================
def main():
    if len(sys.argv) < 4:
        print("用法: python analyze_tone.py <og.wav> <received.wav> <频率Hz>")
        print("示例: python analyze_tone.py 100hz_og.wav 000_2k.wav 100")
        return

    og_path   = sys.argv[1]
    rx_path   = sys.argv[2]
    target_hz = float(sys.argv[3])

    for p in [og_path, rx_path]:
        if not os.path.exists(p):
            print(f"❌ 找不到文件: {p}")
            return

    fs_og, audio_og = load_wav(og_path)
    fs_rx, audio_rx = load_wav(rx_path)

    print(f"OG       : {os.path.basename(og_path)}  fs={fs_og}Hz  时长={len(audio_og)/fs_og:.2f}s")
    print(f"Received : {os.path.basename(rx_path)}  fs={fs_rx}Hz  时长={len(audio_rx)/fs_rx:.2f}s")
    print(f"Target freq: {target_hz} Hz\n")

    og_res = analyze(audio_og, fs_og, target_hz)
    rx_res = analyze(audio_rx, fs_rx, target_hz)

    attenuation = (20 * np.log10(rx_res["peak_amp"] / og_res["peak_amp"])
                   if og_res["peak_amp"] > 0 else float('-inf'))

    print("=" * 50)
    print(f"{'Metric':<20} {'OG':>12} {'Received':>12}")
    print("-" * 50)
    print(f"{'Peak Freq (Hz)':<20} {og_res['peak_freq']:>12.2f} {rx_res['peak_freq']:>12.2f}")
    print(f"{'SNR (dB)':<20} {og_res['snr_db']:>12.1f} {rx_res['snr_db']:>12.1f}")
    print(f"{'Purity (%)':<20} {og_res['purity_pct']:>12.2f} {rx_res['purity_pct']:>12.2f}")
    print(f"{'THD (%)':<20} {og_res['thd_pct']:>12.2f} {rx_res['thd_pct']:>12.2f}")
    print(f"{'Attenuation (dB)':<20} {'—':>12} {attenuation:>12.1f}")
    print("=" * 50)

    plot_comparison(
        {"audio": audio_og, "fs": fs_og},
        og_res,
        {"audio": audio_rx, "fs": fs_rx},
        rx_res,
        target_hz,
    )


if __name__ == "__main__":
    main()
