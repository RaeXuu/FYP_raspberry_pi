# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 系统定位
这是一个**资源效率优先的边缘智能系统**，所有设计决策须以轻量、可靠为目标。

## 设备限制
- **部署设备**：Raspberry Pi 4B（4GB 内存），开发与最终部署均使用同一台设备
- **不要安装不必要的依赖**（尤其是 matplotlib、scipy 等大型库）
- 优先使用 Python 标准库或已有依赖（bleak、numpy、ai_edge_litert、yaml）
- `src/receive.py` 目前使用了 `scipy`，属于遗留代码，不应在新代码中效仿

## 运行命令

```bash
# 激活虚拟环境
source .venv/bin/activate

# 主程序（需要 ESP32 在线）
python main_pi.py

# 离线推理调试（使用本地 WAV 文件，无需硬件）
python main_pi_debug.py

# BLE 调试工具（扫描/检查服务/监控数据流）
python ble_debug.py
```

## 代码架构

### 数据流（在线模式 `main_pi.py`）
```
ESP32 BLE → notification_handler() → bytearray buffer
→ collect_segment()（等待 8000 字节 = 2s）
→ np.int16 转 float32 / 32768
→ preprocess_array_for_pi()（带通滤波 → Log-Mel）
→ TFLite SQA 模型（质量评估，threshold=0.5）
→ TFLite 诊断模型（Normal/Abnormal）
→ SQA 加权平均 → 最终诊断
```

### 预处理流水线（`src/preprocess/`）
| 文件 | 功能 |
|---|---|
| `load_wav.py` | 加载 WAV，重采样到目标采样率 |
| `filters.py` | 带通滤波（25–400 Hz，Butterworth） |
| `segment.py` | 切片（2s，overlap=0.5s） |
| `mel.py` | Log-Mel 频谱，固定输出 (32, 64) |
| `preprocess_pipeline.py` | 组合以上步骤，提供两个入口：`preprocess_wav_for_pi()`（文件输入）和 `preprocess_array_for_pi()`（内存数组输入） |

**TFLite 输入张量格式**：`(1, 1, 32, 64)`，即 `[Batch, Channel, Height, Width]`，dtype=float32。

### 关键参数（`config.yaml`）
- 采样率：2000 Hz，片段时长：2.0s，带通：25–400 Hz
- Mel：n_fft=256, hop=96, n_mels=32, fmin=20, fmax=400

### 模型文件
- `heart_quality_quant.tflite`：SQA 质量评估（index 0=Poor, 1=Good）
- `heart_model_quant.tflite`：心音诊断（index 0=Normal, 1=Abnormal）
- `*_fp32.tflite`：FP32 版本备用，部署时使用 quant 版以节省内存

## BLE 配置
- ESP32 MAC：`80:F1:B2:ED:B4:12`（设备名：ESP32_Steth）
- 特征 UUID：`beb5483e-36e1-4688-b7f5-ea07361b26a8`
- 每包 128 字节，数据格式：16-bit 小端 PCM，采样率 2000 Hz
- 连接后需调用 `client._backend._acquire_mtu()` 以协商最大 MTU

## 采集策略
定量采集 3 个 2s 片段（NUM_COLLECTIONS=3，间隔 COLLECTION_INTERVAL=30s），SQA 过滤低质量片段，对通过的片段做 SQA 加权平均得出最终诊断。结果为 Abnormal 时保存原始音频至 `abnormal_records/`。
