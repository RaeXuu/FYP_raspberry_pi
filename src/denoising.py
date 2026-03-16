import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, medfilt

def cascaded_filter_pipeline(filepath):
    # 1. 加载音频
    fs, data = wavfile.read(filepath)
    if len(data.shape) > 1: data = data[:, 0]
    
    # 转换为 float 并做基础缩放
    data = data.astype(np.float32) / 32768.0 

    # 2. 中值滤波：去除由于传感器摩擦产生的脉冲刺 [基于你的项目需求]
    data_med = medfilt(data, kernel_size=5)

    # 3. 级联滤波 (Cascaded Filtering) - 严格复刻文献逻辑
    # 第一步：3阶高通 (4Hz) 专门杀掉呼吸和身体晃动的“大波浪”
    sos_hp = butter(3, 4, btype='high', fs=fs, output='sos')
    data_hp = sosfilt(sos_hp, data_med)
    
    # # 第二步：2阶带通 (20-220Hz) 提取心音核心分量
    # sos_bp = butter(2, [20, 220], btype='band', fs=fs, output='sos')
    # cleaned = sosfilt(sos_bp, data_hp)
    cleaned = data_hp


    # 4. 归一化 (Rescaling)：让波形在视觉和后续 AI 训练中保持一致
    # 采用最大值缩放，并留出 10% 的余量防止视觉过载
    norm_data = cleaned / (np.max(np.abs(cleaned)) + 1e-6)

    return norm_data, fs

# --- 运行与可视化 ---
filename = "/mnt/d/FypProj/esp32_debug/heart_sound_1770104436.wav" # 确保文件在你的 VS Code 工作目录下
try:
    final_signal, fs = cascaded_filter_pipeline(filename)
    time = np.linspace(0, len(final_signal)/fs, len(final_signal))

    plt.figure(figsize=(12, 8))

    # 子图1：精细时域波形
    plt.subplot(2, 1, 1)
    # 聚焦显示其中的一段（例如 2-7秒），方便看清 $S1/S2$ 结构
    plt.plot(time, final_signal, color='#007acc', linewidth=0.8)
    plt.xlim(2, 7) 
    plt.ylim(-1.1, 1.1)
    plt.title(f"Refined Pipeline: 3rd-order HP (4Hz) + 2nd-order BP (20-220Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 子图2：语谱图 (使用 magma 颜色展示能量分布)
    plt.subplot(2, 1, 2)
    plt.specgram(final_signal, NFFT=2048, Fs=fs, noverlap=1024, cmap='magma')
    plt.ylim(0, 500) # 只看 500Hz 以下的心音关键带
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label='Intensity (dB)')

    plt.tight_layout()
    plt.show()

    # 保存处理后的 WAV 文件
    base, ext = os.path.splitext(filename)
    output_filename = base + "_processed" + ext
    output_int16 = (final_signal * 32767).astype(np.int16)
    wavfile.write(output_filename, fs, output_int16)
    print(f"已保存处理后文件：{output_filename}")

except FileNotFoundError:
    print(f"找不到文件 {filename}，请检查路径。")