# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 系统定位
这是一个**资源效率优先的边缘智能系统**，所有设计决策须以轻量、可靠为目标。
这是一个部署在边缘设备上的心音诊断系统。

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
→ 积累 80000 字节（20s = 40000 samples × 2 bytes）→ chunk_queue
→ inference_worker() 消费队列
→ 带通滤波整块一次
→ 滑窗切片（2s 窗口，1s hop）→ per-window 峰值归一化
→ logmel_fixed_size()（Log-Mel，shape=(64,64)）
→ TFLite SQA 模型（质量评估，SQA_THRESHOLD=0.6）
→ TFLite 诊断模型（Normal/Abnormal）
→ SQA 加权平均 → 块级 label
→ 所有块保存至 debug_records/（normal_/abnormal_/noise_ 前缀）
→ append_summary() 写 records/summary.jsonl
```

### 预处理流水线（`src/preprocess/`）
| 文件 | 功能 |
|---|---|
| `load_wav.py` | 加载 WAV，重采样到目标采样率 |
| `filters.py` | 带通滤波（25–400 Hz，Butterworth） |
| `segment.py` | 切片（2s，overlap=0.5s） |
| `mel.py` | Log-Mel 频谱，固定输出 (64, 64) |
| `preprocess_pipeline.py` | 组合以上步骤，提供两个入口：`preprocess_wav_for_pi()`（文件输入）和 `preprocess_array_for_pi()`（内存数组输入） |

**TFLite 输入张量格式**：`(1, 1, 64, 64)`，即 `[Batch, Channel, Height, Width]`，dtype=float32。

### 关键参数（`config.yaml`）
- 采样率：2000 Hz，片段时长：2.0s，带通：25–400 Hz
- Mel：n_fft=256, win_length=256, hop_length=128, n_mels=64, fmin=25, fmax=400, target_frames=64

### 模型文件
- `heart_quality_quant.tflite`：SQA 质量评估（index 0=Poor, 1=Good）
- `heart_model_quant.tflite`：心音诊断（index 0=Normal, 1=Abnormal）
- `*_fp32.tflite`：FP32 版本备用，部署时使用 quant 版以节省内存

## OLED 配置

| 屏幕 | 类 | I2C bus | 驱动 | 尺寸 | 内容 |
|---|---|---|---|---|---|
| OLED 1（主屏） | `OLEDDisplay` | port=1（GPIO2/3） | ssd1306 128×64 | 诊断状态、连接进度、推理结果 |
| OLED 2（副屏） | `SysInfoDisplay` | port=4（GPIO23/24） | ssd1306 128×32 rotate=0 | CPU%、内存、温度（每2s刷新） |

- OLED 2 用软件 I2C：`dtoverlay=i2c-gpio,bus=4,i2c_gpio_sda=23,i2c_gpio_scl=24`
- **OLED 2 必须用 `ssd1306(width=128, height=32, rotate=0)`**，`sh1106` 或 `height=64` 均显示异常

## BLE 配置
- ESP32 MAC：`80:F1:B2:ED:B4:12`（设备名：ESP32_Steth）
- 特征 UUID：`beb5483e-36e1-4688-b7f5-ea07361b26a8`
- 每包 128 字节，数据格式：16-bit 小端 PCM，采样率 2000 Hz
- 连接后需调用 `client._backend._acquire_mtu()` 以协商最大 MTU

## 采集策略
连续流式采集，每 **20s** 为一块（`CHUNK_DURATION=20`）。每块内做滑动窗口推理（2s 窗口，1s hop，共约 19 个窗口）。SQA 过滤低质量窗口（`SQA_THRESHOLD=0.65`），对通过的窗口做 SQA 加权平均得出块级诊断。所有块（Normal/Abnormal/低质量）均以对应前缀保存至 `debug_records/`；每块结果写入 `records/summary.jsonl`。短按按键停止当前会话，再次短按启动新会话。
