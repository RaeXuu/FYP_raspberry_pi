# 嵌入式系统模块规划

本文档梳理整个心音诊断系统的完整模块构成，覆盖 ESP32（采集端）与 Raspberry Pi Zero 2W（推理端）两侧，以及系统集成所需的各类基础设施。

---

## 系统总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        用户                                       │
│            按键触发 ←──── LED/OLED 状态 ←──── 蜂鸣器提示         │
└───────────────────┬──────────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────┐
│                Raspberry Pi Zero 2W（推理端）                     │
│                                                                   │
│  [BLE接收] → [预处理] → [TFLite推理] → [结果管理] → [上报]      │
│                                                                   │
│  [系统服务] [看门狗] [电源管理] [日志] [数据清理]                │
└───────────────────┬──────────────────────────────────────────────┘
                    │ BLE (2.4 GHz)
┌───────────────────▼──────────────────────────────────────────────┐
│                   ESP32（采集端）                                  │
│                                                                   │
│  [麦克风/ADC] → [硬件滤波] → [PCM缓冲] → [BLE GATT Server]      │
│  [按键] [状态LED] [电池/USB供电]                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 模块一：ESP32 采集端

> 已实现大部分，此处梳理完整边界。

### 1.1 音频采集
- **硬件**：模拟驻极体麦克风 + 运放 → ESP32 ADC（GPIO34）
- **参数**：采样率 2000 Hz，12-bit ADC，软件截断为 16-bit PCM（小端）
- **缓冲**：双缓冲（ping-pong）以避免采样空洞，每包 128 字节通过 BLE notification 发出

### 1.2 BLE GATT Server
- **服务 UUID**：自定义
- **特征 UUID**：`beb5483e-36e1-4688-b7f5-ea07361b26a8`（Notify 属性）
- **MTU 协商**：启动后主动请求最大 MTU，减少分包开销
- **连接管理**：断连后自动重启广播，等待 Pi 重连

### 1.3 ESP32 电源管理
- **工作模式**：采集时全速运行；空闲（未连接）时进入 Light Sleep
- **唤醒源**：BLE 连接事件 / 物理按键
- **供电方案**：USB-C 或 3.7V 锂电池 + TP4056 充电模块

### 1.4 ESP32 状态指示
| LED 颜色 | 含义 |
|---|---|
| 蓝色慢闪 | 广播中，等待连接 |
| 蓝色常亮 | BLE 已连接 |
| 绿色快闪 | 正在采集传输 |
| 红色 | 错误 / 电量低 |

---

## 模块二：Pi 端 — 算法核心（已实现）

> 详见 `CLAUDE.md`，此处仅列边界。

- `src/preprocess/`：带通滤波 → Log-Mel 频谱
- `heart_quality_quant.tflite`：SQA 质量评估
- `heart_model_quant.tflite`：Normal / Abnormal 诊断
- `main_pi.py`：BLE 接收 + 流式推理主循环

---

## 模块三：用户交互模块

> **目标**：无屏幕、无键盘也能独立操作，适合便携场景。

### 3.1 物理按键
| 按键 | 短按 | 长按（3s） |
|---|---|---|
| BTN_START | 触发一次完整采集（3块×2s） | 关机 |
| BTN_RESULT | 播报/显示最近一次结果 | 清除历史记录 |

- 接 GPIO，使用内部上拉，软件消抖（5ms）
- 实现文件：`src/ui/button.py`

### 3.2 LED 状态指示
| 状态 | 指示 |
|---|---|
| 空闲 | 绿色慢闪（1Hz） |
| BLE 连接中 | 蓝色快闪（4Hz） |
| 采集中 | 蓝色常亮 |
| 推理中 | 黄色呼吸 |
| 结果 Normal | 绿色亮 3s |
| 结果 Abnormal | 红色亮 5s + 蜂鸣 |
| 错误 | 红色快闪 |

- 使用单颗 RGB LED（共阴），GPIO PWM 驱动
- 实现文件：`src/ui/led.py`

### 3.3 蜂鸣器（可选）
- 无源蜂鸣器，GPIO PWM 控制频率
- 仅在结果 Abnormal 或系统错误时鸣响，不干扰采集
- 实现文件：`src/ui/buzzer.py`

### 3.4 OLED 显示屏（可选扩展）
- SSD1306 128×32，I2C 接口
- 显示内容：当前状态、最近结果、电量
- 库：`luma.oled`（若内存允许），或直接操作 I2C 帧缓冲
- **Zero 2W 内存紧张时可去掉此模块**

---

## 模块四：结果管理与存储

### 4.1 结构化日志（已有基础）
- `debug_records/inference_log.csv`：每窗口 SQA 分数、诊断概率
- 扩展为按日期分目录：`records/YYYY-MM-DD/`

### 4.2 异常音频存档（已有基础）
- Abnormal 结果自动保存原始 PCM → WAV
- 存档路径：`abnormal_records/`

### 4.3 结果摘要文件
- 每次完整采集后写入 `records/summary.jsonl`，一行一条记录：
  ```json
  {"ts": "2026-03-29T10:30:00", "label": "Normal", "prob_normal": 0.82, "valid_segs": 3, "total_segs": 3}
  ```
- 便于后续统计和上报，不依赖 CSV 解析

### 4.4 数据清理策略
- 本地最多保留 30 天数据，超期自动删除
- WAV 文件大小上限：50MB（Zero 2W SD 卡容量有限）
- 实现文件：`src/storage/cleaner.py`，由 systemd timer 每日触发

---

## 模块五：网络上报模块（可选）

> Zero 2W 有板载 WiFi，可在有网环境下上报结果。

### 5.1 上报触发时机
- 每次采集完成后尝试上报（非阻塞，失败不影响本地记录）
- 上报失败时写入待发队列，WiFi 恢复后批量发送

### 5.2 上报协议
- **HTTP POST**（简单场景）：将 `summary.jsonl` 最新条目 POST 到后端
- **MQTT**（物联网场景）：主题 `heart/{device_id}/result`，QoS=1

### 5.3 隐私与安全
- 上报内容仅含结果摘要（标签、概率、时间戳），**不上传原始音频**
- HTTPS/TLS，设备端持有证书或预共享密钥
- 实现文件：`src/network/reporter.py`

---

## 模块六：系统服务与可靠性

### 6.1 开机自启（systemd）
```ini
# /etc/systemd/system/heartbeat.service
[Unit]
Description=Heart Sound Diagnostic Service
After=bluetooth.target network.target

[Service]
User=pi
WorkingDirectory=/home/pi/FypPi
ExecStart=/home/pi/FypPi/.venv/bin/python main_pi.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```
- 文件：`deploy/heartbeat.service`

### 6.2 软件看门狗
- 主循环每 30s 写一次心跳时间戳到 `/tmp/heartbeat.ts`
- 独立看门狗进程检查时间戳，超时 90s 则重启主服务
- 实现文件：`src/watchdog.py`
- 也可直接启用 Linux 内核硬件看门狗：`/dev/watchdog`

### 6.3 日志管理
- 使用 Python `logging` 模块，级别 INFO，文件 `logs/app.log`
- `RotatingFileHandler`：单文件最大 1MB，保留 3 份（Zero 2W 省空间）
- 错误级别同时打印到 stderr（便于 journald 捕获）

### 6.4 异常恢复
| 异常类型 | 处理策略 |
|---|---|
| BLE 断连 | 自动重连，指数退避（1s → 2s → 4s，上限 30s） |
| TFLite 推理失败 | 跳过当前块，记录错误，继续下一块 |
| 队列积压 | 丢弃最旧块，保留最新数据（已实现） |
| 内存耗尽 | OOM killer 触发前由看门狗重启 |

---

## 模块七：部署与更新

### 7.1 代码部署
```bash
# Zero 2W 上执行
git pull origin main
sudo systemctl restart heartbeat
```
- Zero 2W 本身不开发，通过 git 从 Pi 4B 同步

### 7.2 模型更新
- 新模型文件放入项目根目录，命名不变（`*_quant.tflite`）
- 重启服务即可加载新模型，无需修改代码

### 7.3 配置更新
- 所有可调参数集中在 `config.yaml`，修改后重启服务生效
- 不要把 MAC 地址等设备相关配置硬编码进代码

---

## 模块八：电源管理（Pi 端）

### 8.1 供电方案
- **有线**：5V/2.5A USB-C（开发调试）
- **便携**：5V 移动电源（UPS HAT 可实现不间断供电）
- Zero 2W 峰值功耗约 1.3W，正常工作约 0.7W

### 8.2 软件节能
- 推理空闲时（等待 BLE 数据）：asyncio 事件循环自然休眠，CPU 占用低
- 禁用不需要的外设（HDMI、USB hub）：写入 `/boot/config.txt`
  ```
  dtoverlay=disable-wifi   # 无需WiFi时
  hdmi_blanking=2
  ```
- 可选：采集完成后通过 `systemctl suspend` 进入待机，按键唤醒

### 8.3 安全关机
- 捕获 SIGTERM / 按键长按 → 优雅停止 BLE → flush 日志 → `shutdown -h now`
- 防止 SD 卡写入中断导致文件系统损坏

---

## 当前实现状态

| 模块 | 状态 | 备注 |
|---|---|---|
| ESP32 BLE 采集 | 已实现 | — |
| BLE 接收（Pi 端） | 已实现 | `main_pi.py` |
| 预处理流水线 | 已实现 | `src/preprocess/` |
| TFLite 双模型推理 | 已实现 | SQA + 诊断 |
| CSV 推理日志 | 已实现 | `debug_records/` |
| 物理按键 | 未实现 | `src/ui/button.py` |
| LED 状态指示 | 未实现 | `src/ui/led.py` |
| 蜂鸣器 | 未实现 | `src/ui/buzzer.py` |
| OLED 显示 | 未实现（可选） | 内存允许再加 |
| 结果摘要 JSONL | 未实现 | `src/storage/` |
| 数据清理 | 未实现 | `src/storage/cleaner.py` |
| 网络上报 | 未实现（可选） | `src/network/reporter.py` |
| systemd 服务 | 未实现 | `deploy/heartbeat.service` |
| 软件看门狗 | 未实现 | `src/watchdog.py` |
| 安全关机 | 部分（SIGINT） | 需扩展按键长按 |
