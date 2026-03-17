# 管道问题记录

---

## 当前完整 Pipeline（main_pi.py）

### 第一层：BLE 接收后的格式转换（main_pi.py 自己做）

这部分与模型训练无关，是为了把 ESP32 的原始数据转成可处理的格式。

```python
raw = await collect_segment()
# BLE 持续接收字节，直到凑够 SEGMENT_BYTES = 8000 × 2字节 × 2秒 = 32000 字节
# 注意：这个数字与 ESP32 采样率强绑定，ESP32 采样率改了这里必须跟着改

audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
# 字节 → int16 整数 → float32，缩放到 ±1.0

audio = audio[::4]
# 跳采样 8000Hz → 2000Hz（⚠️ 见下方问题1）
```

### 第二层：preprocess_array_for_pi()（与训练数据处理一致）

```python
apply_bandpass(audio, 25–400 Hz)   # 带通滤波，去除心音范围外频率
  ↓ logmel_fixed_size(...)          # Log-Mel 频谱图
  ↓ fix_length → (32, 64)           # 固定尺寸
  ↓ reshape → (1, 1, 32, 64)        # 适配 TFLite 输入格式
```

---

## preprocess_pipeline 两个函数的区别

| | `preprocess_wav_for_pi` | `preprocess_array_for_pi` |
|---|---|---|
| 输入 | WAV 文件路径 | numpy 数组（已是 2000Hz） |
| 重采样 | load_wav 内部自动处理 | 不做，依赖调用方传入正确 sr |
| 切片 | segment_audio 切成多个 2s 片段 | 不切，默认传入已经是 2s |
| 返回 | list of tensors | 单个 tensor |

`preprocess_array_for_pi` 没有切片不是 bug，因为 `collect_segment()` 已经在外面精确收集了 2s 数据。

---

## 待修复问题

### 问题1：跳采样导致混叠失真

**涉及文件：**
- `main_pi.py` 第 115 行
- `record_debug.py` 第 144 行

**当前代码：**
```python
audio = audio[::4]  # 8000Hz → 2000Hz
```

**问题：** 直接跳采样没有先做低通滤波，1000Hz 以上的高频成分会折叠回低频污染心音信号。训练数据直接读取 2000Hz WAV，不存在这一步，造成训练和推理不一致。

**修复方向：** 用 `scipy.signal.decimate` 或 `scipy.signal.resample_poly` 替换，两者会自动先低通再降采样。注意 CLAUDE.md 要求尽量避免 scipy，可评估是否用其他方式。

### 问题2：采样率硬编码分散

`8000`（ESP32 采样率）、`2000`（模型输入采样率）、`[::4]`（降采样倍数）分散在多个文件中硬编码，改一处容易漏改其他地方。建议统一放入 `config.yaml` 管理。
