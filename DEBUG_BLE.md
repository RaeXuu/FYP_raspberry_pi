# BLE 连接调试参考文档

## 一、ESP32 端已知参数（来源：arduino.c）

### 音频采集
| 参数 | 值 | 说明 |
|---|---|---|
| 采样率 | 8000 Hz | I2S SAMPLE_RATE |
| 位深 | 16-bit | I2S_BITS_PER_SAMPLE_16BIT |
| 声道 | 单声道（左） | I2S_CHANNEL_FMT_ONLY_LEFT |
| DMA 缓冲区数量 | 8 | dma_buf_count |
| DMA 缓冲区长度 | 128 字节 | dma_buf_len |

### 信号处理
| 参数 | 值 | 说明 |
|---|---|---|
| 低通滤波系数 α | 0.05 | 保留 20–400 Hz 心音 |
| 数字增益 | 30× | digital_gain |
| 限幅范围 | ±32000 | 16-bit 保护 |
| 直流去除窗口 | 1000 样本 | 滑动平均 |

### BLE 传输
| 参数 | 值 | 说明 |
|---|---|---|
| 单包大小 | 128 字节 | BLOCK_SIZE |
| 每包样本数 | 64 samples | 128 / 2字节 |
| 理论包速率 | 125 包/秒 | 8000 / 64 |
| 理论带宽 | 15.625 KB/s | 125 × 128 |
| **实测带宽** | **14.32 KB/s** | ble_debug 模式3 实测（3752包，469KB，30s） |
| BLE 传输开销 | ~8% | 理论 15.625 KB/s vs 实测 14.32 KB/s |

### BLE 连接参数（ESP32 向 Pi 请求）
| 参数 | 原始值 | 换算 | 说明 |
|---|---|---|---|
| 连接间隔（最小） | 6 | 7.5 ms | × 1.25ms |
| 连接间隔（最大） | 12 | 15 ms | × 1.25ms |
| Slave Latency | 0 | — | 不跳包 |
| Supervision Timeout | 100 | **1000 ms** | × 10ms，超时即断连 |

> ⚠️ Supervision Timeout 仅 1 秒：若 Pi 端事件循环阻塞超过 1 秒（如推理耗时过长），连接会被强制断开。

### BLE 标识符
| 参数 | 值 |
|---|---|
| 设备名 | ESP32_Steth |
| MAC 地址 | 80:F1:B2:ED:B4:12 |
| Service UUID | 4fafc201-1fb5-459e-8fcc-c5c9c331914b |
| Characteristic UUID | beb5483e-36e1-4688-b7f5-ea07361b26a8 |
| Characteristic 属性 | READ / WRITE / NOTIFY / INDICATE |

### 派生计算（main_pi.py 相关）
| 参数 | 值 | 说明 |
|---|---|---|
| 每段所需字节数 | 32000 字节 | 8000Hz × 2字节 × 2秒 |
| 实际采集耗时 | ~2.18 秒 | 32000 / 14663（14.32×1024） |
| 降采样后样本数 | 4000 samples | 送入模型的 2000Hz × 2s |

---

## 二、Pi 端已知参数（来源：ble_debug 模式2 / 实测）

### GATT 服务结构
| 顺序 | 服务 | UUID | 说明 |
|---|---|---|---|
| 1 | Generic Access Profile | 00001800-... | 标准服务 |
| 2 | Generic Attribute Profile | 00001801-... | 标准服务 |
| 3 | 自定义音频服务 | 4fafc201-... | ESP32 心音服务 |

### 特征列表
| 特征 | UUID | 属性 | 当前值 |
|---|---|---|---|
| Device Name | 00002a00-... | read | `45535033325f5374657468`（11字节，"ESP32_Steth"） |
| Appearance | 00002a01-... | read | `0000`（2字节，未定义外观） |
| Central Address Resolution | 00002aa6-... | read | `00`（1字节，不支持） |
| Service Changed | 00002a05-... | indicate | — |
| **目标音频特征** | beb5483e-... | read/write/notify/indicate | 128字节实时 PCM 数据（读到当前帧） |

### MTU
| 场景 | MTU 值 | 说明 |
|---|---|---|
| 未调用 `_acquire_mtu()` | 23 字节（默认） | BLE ATT 最小值（模式2实测，bleak 会发 UserWarning） |
| 调用 `_acquire_mtu()` 后 | **517 字节** | 模式3 / main_pi.py 实测 |

### 发现的问题
| 问题 | 现象 | 根因 |
|---|---|---|
| EOFError | 每次断开连接时抛出 | `dbus_fast` 库 bug：BlueZ 先关闭 D-Bus socket，disconnect() 读到 EOF |

> ⚠️ `main_pi.py` 中的 `EOFError` 与此相同——数据收发本身成功，错误发生在事件循环恢复后尝试 D-Bus 操作时（可能是 `stop_notify` 或 `disconnect`）。

---

## 三、HCI 层参数（来源：sudo btmon + ble_debug 模式3）

### 实际协商连接参数（每次连接均一致）
| 参数 | HCI 实测值 | 说明 |
|---|---|---|
| Connection Interval | **15.00 ms (0x000c)** | ESP32 请求最大值（12 × 1.25ms），Pi 直接接受 |
| Connection Latency | 0 | 不跳包 |
| Supervision Timeout | **1000 ms (0x0064)** | 与 `updateConnParams(..., 100)` 一致 |

### ESP32 声明的 LE 特性（Feature Exchange 成功时）
```
Features: 0x3f 0x00 0x00 0x08
  LE Encryption
  Connection Parameter Request Procedure
  Extended Reject Indication
  Peripheral-initiated Features Exchange
  LE Ping
  LE Data Packet Length Extension
  Remote Public Key Validation
```

### 连接建立阶段：Feature Exchange 0x3e 重试现象
| 项目 | 值 |
|---|---|
| 成功前重试次数 | **~35 次**（时间戳 17s → 26s，约 9 秒） |
| 失败阶段 | `LE Read Remote Used Features` 返回 `0x3e` |
| 每次失败耗时 | ~100–300 ms（HCI 连接成功 → Feature Exchange 失败 → 断开） |
| 最终成功 | 第 36 次，时间戳 ~26.79s，Feature Exchange 返回 `Success` |

> **现象解读**：HCI 层连接（Connection Complete）每次都成功，但 Feature Exchange 随机失败并触发断开，bleak 自动重试。与 ESP32 的 Supervision Timeout（1s）无直接关系（失败发生在 ~100ms 内）。可能是 ESP32 主循环忙于 I2S 读写而未能及时响应 Feature Exchange。成功建立后连接稳定，30 秒内无异常断开。
