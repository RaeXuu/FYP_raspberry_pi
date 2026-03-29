# 管道问题记录

---

## 当前完整 Pipeline（main_pi.py）

### 第一层：BLE 接收后的格式转换（main_pi.py 自己做）

```python
raw = await collect_segment()
# BLE 持续接收字节，直到凑够 SEGMENT_BYTES = 2000 × 2字节 × 2秒 = 8000 字节
# ESP32 已在硬件侧完成抗混叠 LPF（截止 ~800Hz）+ 4:1 抽取（8000Hz → 2000Hz）
# Pi 端收到的已经是 2000Hz 数据，不需要再降采样

audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
# 字节 → int16 整数 → float32，缩放到 ±1.0
```

### 第二层：preprocess_array_for_pi()（与训练数据处理一致）

```python
peak_normalize(audio)              # 峰值归一化到 ±1.0
  ↓ apply_bandpass(25–400 Hz)      # 带通滤波，去除心音范围外频率
  ↓ segment_audio()                # 切片（2s，overlap=0.5s），不足补零
  ↓ logmel_fixed_size(...)         # 每片 → Log-Mel 频谱图
  ↓ reshape → (1, 1, 32, 64)       # 适配 TFLite 输入格式
  ↓ 返回 list of tensors
```

### 第三层：推理循环

```python
for tensor in tensors:             # 遍历所有切片
    SQA 质量评估 → sqa_score
    if sqa_score < SQA_THRESHOLD: continue
    诊断模型 → prob_normal
    results.append((sqa_score, prob_normal))

# SQA 加权平均 → 最终诊断
```

---

## preprocess_pipeline 两个函数的区别

| | `preprocess_wav_for_pi` | `preprocess_array_for_pi` |
|---|---|---|
| 输入 | WAV 文件路径 | numpy 数组（已是 2000Hz） |
| 重采样 | load_wav 内部自动处理 | 不做，依赖调用方传入正确 sr |
| 归一化 | load_wav 内部处理 | 峰值归一化（max=1.0） |
| 切片 | segment_audio 切成多个 2s 片段 | segment_audio 切成多个 2s 片段，不足补零 |
| 返回 | list of tensors | list of tensors |

两个函数现在行为完全对齐。

---

## record_debug.py 行为说明

- 录音：通过 BLE 接收原始 2000Hz PCM
- 保存 WAV：**峰值归一化后**保存到 `WAV_record/`，与推理时听感一致（原始信号无增益，直接保存幅值极小）
- 推理：走与 main_pi.py 完全相同的 `preprocess_array_for_pi` 流程

---

## 已解决问题

### ~~问题1：跳采样导致混叠失真~~（已解决）

原 `main_pi.py` 用 `audio[::4]` 直接跳采样 8000Hz→2000Hz，没有先做低通滤波。

**解决方式**：ESP32 端 arduino.c 已改为硬件侧做抗混叠 LPF（2阶 IIR，截止 ~800Hz）+ 4:1 抽取，Pi 端收到的已经是干净的 2000Hz 信号，`audio[::4]` 已删除。

### ~~问题2：采样率硬编码分散~~（部分解决）

`SAMPLE_RATE=2000` 现在在 `main_pi.py` 和 `record_debug.py` 顶部统一定义，`config.yaml` 也有对应字段。ESP32 端采样率仍在 `arduino.c` 中硬编码。

---

## 待关注问题

### 问题3：ESP32 无数字增益

arduino.c 目前没有数字增益，INMP441 原始信号较弱，存下来的 WAV 幅值很小（SNR 实测约 -28dB）。`preprocess_array_for_pi` 的峰值归一化会把信号拉满，推理不受影响，但说明信号接近量化噪声底限，实际心音场景下需关注。

