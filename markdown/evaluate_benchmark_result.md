"""

TFLite 模型性能基准测试（树莓派端）

测试 6 个模型的推理延迟：FP32 / INT8 动态 / INT8 全整型 × SQA / Diagnosis

  

使用方式

python benchmark.py # 默认 100 轮，10 轮预热

python benchmark.py --runs 200 # 自定义轮数

python benchmark.py --warmup 20 # 自定义预热轮数

python benchmark.py --model diag # 只测诊断模型

python benchmark.py --model sqa # 只测 SQA 模型

"""




"""

量化模型准确率评估（Pi 端）

对测试集每条音频跑完整推理流水线，对比 FP32 vs INT8 动态 vs INT8 全整型。

  

模型输出索引约定

SQA 模型（heart_quality_*.tflite）：

训练时 label 0 = Good, label 1 = Bad（reversed convention）

→ index 0 = Good 概率, index 1 = Bad 概率

  

诊断模型（heart_model_*.tflite）：

训练时 label 0 = Normal, label 1 = Abnormal

→ index 0 = Normal 概率, index 1 = Abnormal 概率

  

使用方式

python evaluate.py --mode sqa # SQA 独立评估

python evaluate.py --mode diag # 诊断模型（无 SQA 门控）

python evaluate.py --mode both # 耦合流水线（SQA 门控 + 加权）

python evaluate.py --mode sqa --verify # 额外打印若干样本输出，排查索引

python evaluate.py --mode all # 全部模式依次执行

"""



**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py

============================================================

  TFLite 模型性能基准测试

  TFLite 后端: ai_edge_litert

  输入形状:   (1, 1, 64, 64)

  预热轮数:   10

  测试轮数:   100

  ────────────────────────────────────────────────

  芯片:       Raspberry Pi 4 Model B Rev 1.5

  CPU 温度:   36.5°C

  CPU 频率:   1800 MHz

  内存:       3172 MB 可用 / 3797 MB 总量

============================================================

  

────────────────────────────────────────────────────────────

  模型: Diag FP32

  文件: /home/rasp4b/FypPi/heart_model_fp32.tflite

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  [load] heart_model_fp32.tflite       in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=14.49ms  median=14.12ms  p95=16.03ms  min=13.83ms  max=21.61ms  std=1.47ms

  

────────────────────────────────────────────────────────────

  模型: Diag INT8动态

  文件: /home/rasp4b/FypPi/heart_model_quant.tflite

  [load] heart_model_quant.tflite      in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=14.03ms  median=13.99ms  p95=14.31ms  min=13.84ms  max=14.56ms  std=0.13ms

  

────────────────────────────────────────────────────────────

  模型: Diag INT8全整型

  文件: /home/rasp4b/FypPi/heart_model_int8full.tflite

  [load] heart_model_int8full.tflite   in=int8     out=int8     in_scale=0.325044  out_scale=0.069440

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=8.83ms  median=8.72ms  p95=8.80ms  min=8.69ms  max=18.75ms  std=1.00ms

  

────────────────────────────────────────────────────────────

  模型: SQA FP32

  文件: /home/rasp4b/FypPi/heart_quality_fp32.tflite

  [load] heart_quality_fp32.tflite     in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=14.43ms  median=14.40ms  p95=14.71ms  min=14.23ms  max=14.96ms  std=0.15ms

  

────────────────────────────────────────────────────────────

  模型: SQA INT8动态

  文件: /home/rasp4b/FypPi/heart_quality_quant.tflite

  [load] heart_quality_quant.tflite    in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=13.88ms  median=13.84ms  p95=14.12ms  min=13.65ms  max=14.54ms  std=0.16ms

  

────────────────────────────────────────────────────────────

  模型: SQA INT8全整型

  文件: /home/rasp4b/FypPi/heart_quality_int8full.tflite

  [load] heart_quality_int8full.tflite  in=int8     out=int8     in_scale=0.324692  out_scale=0.042427

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=8.67ms  median=8.65ms  p95=8.75ms  min=8.63ms  max=8.80ms  std=0.04ms

  

========================================================================

  诊断模型 (Diagnosis) 三模型推理延迟对比 (median, ms)

========================================================================

  指标                            FP32        INT8动态       INT8全整型

  --------------------------------------------------------

  Median (ms)                 14.12         13.99          8.72 

  Speedup vs FP32              1.00x        1.01x         1.62x 

  --------------------------------------------------------

  P95 (ms)                    16.03         14.31          8.80 

  Std (ms)                     1.47          0.13          1.00 

========================================================================

  

========================================================================

  SQA 质量检测模型 三模型推理延迟对比 (median, ms)

========================================================================

  指标                            FP32        INT8动态       INT8全整型

  --------------------------------------------------------

  Median (ms)                 14.40         13.84          8.65 

  Speedup vs FP32              1.00x        1.04x         1.66x 

  --------------------------------------------------------

  P95 (ms)                    14.71         14.12          8.75 

  Std (ms)                     0.15          0.16          0.04 

========================================================================

  

============================================================

  模型文件大小

============================================================

  模型                                大小 (KB)

  ────────────────────────────────────────

  Diag FP32                          302.8

  Diag INT8动态                        144.7

  Diag INT8全整型                       144.4

  SQA FP32                           302.8

  SQA INT8动态                         144.7

  SQA INT8全整型                        144.4

  

  总耗时: 8.4s

============================================================

**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py

============================================================

  TFLite 模型性能基准测试

  TFLite 后端: ai_edge_litert

  输入形状:   (1, 1, 64, 64)

  预热轮数:   10

  测试轮数:   100

  ────────────────────────────────────────────────

  芯片:       Raspberry Pi 4 Model B Rev 1.5

  CPU 温度:   37.5°C

  CPU 频率:   1800 MHz

  内存:       3172 MB 可用 / 3797 MB 总量

============================================================

  

────────────────────────────────────────────────────────────

  模型: Diag FP32

  文件: /home/rasp4b/FypPi/heart_model_fp32.tflite

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  [load] heart_model_fp32.tflite       in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=14.22ms  median=14.17ms  p95=14.52ms  min=13.99ms  max=15.06ms  std=0.19ms

  

────────────────────────────────────────────────────────────

  模型: Diag INT8动态

  文件: /home/rasp4b/FypPi/heart_model_quant.tflite

  [load] heart_model_quant.tflite      in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=13.70ms  median=13.67ms  p95=14.00ms  min=13.50ms  max=14.23ms  std=0.13ms

  

────────────────────────────────────────────────────────────

  模型: Diag INT8全整型

  文件: /home/rasp4b/FypPi/heart_model_int8full.tflite

  [load] heart_model_int8full.tflite   in=int8     out=int8     in_scale=0.325044  out_scale=0.069440

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=8.67ms  median=8.66ms  p95=8.78ms  min=8.63ms  max=8.84ms  std=0.04ms

  

────────────────────────────────────────────────────────────

  模型: SQA FP32

  文件: /home/rasp4b/FypPi/heart_quality_fp32.tflite

  [load] heart_quality_fp32.tflite     in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=14.00ms  median=13.96ms  p95=14.31ms  min=13.83ms  max=14.57ms  std=0.14ms

  

────────────────────────────────────────────────────────────

  模型: SQA INT8动态

  文件: /home/rasp4b/FypPi/heart_quality_quant.tflite

  [load] heart_quality_quant.tflite    in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=13.81ms  median=13.76ms  p95=14.24ms  min=13.61ms  max=14.39ms  std=0.17ms

  

────────────────────────────────────────────────────────────

  模型: SQA INT8全整型

  文件: /home/rasp4b/FypPi/heart_quality_int8full.tflite

  [load] heart_quality_int8full.tflite  in=int8     out=int8     in_scale=0.324692  out_scale=0.042427

  预热 10 轮... 完成

  基准测试 100 轮... 完成

  mean=8.71ms  median=8.70ms  p95=8.76ms  min=8.67ms  max=8.82ms  std=0.03ms

  

========================================================================

  诊断模型 (Diagnosis) 三模型推理延迟对比 (median, ms)

========================================================================

  指标                            FP32        INT8动态       INT8全整型

  --------------------------------------------------------

  Median (ms)                 14.17         13.67          8.66 

  Speedup vs FP32              1.00x        1.04x         1.64x 

  --------------------------------------------------------

  P95 (ms)                    14.52         14.00          8.78 

  Std (ms)                     0.19          0.13          0.04 

========================================================================

  

========================================================================

  SQA 质量检测模型 三模型推理延迟对比 (median, ms)

========================================================================

  指标                            FP32        INT8动态       INT8全整型

  --------------------------------------------------------

  Median (ms)                 13.96         13.76          8.70 

  Speedup vs FP32              1.00x        1.01x         1.60x 

  --------------------------------------------------------

  P95 (ms)                    14.31         14.24          8.76 

  Std (ms)                     0.14          0.17          0.03 

========================================================================

  

============================================================

  模型文件大小

============================================================

  模型                                大小 (KB)

  ────────────────────────────────────────

  Diag FP32                          302.8

  Diag INT8动态                        144.7

  Diag INT8全整型                       144.4

  SQA FP32                           302.8

  SQA INT8动态                         144.7

  SQA INT8全整型                        144.4

  

  总耗时: 8.2s

============================================================












**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python evaluate.py --mode all

  

============================================================

SQA 模型评估（test_split_sqa.csv）

  测试录音数：324  Bad(label=0)=32  Good(label=1)=292

  索引约定：SQA_IDX_BAD=1  SQA_IDX_GOOD=0

============================================================

  

  [FP32]  SQA=heart_quality_fp32.tflite

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  [load] heart_quality_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

    FP32: 100%|█████████████████████████████████████████████████████████████████████| 324/324 [02:22<00:00,  2.28file/s]

    TP=435 TN=4607 FP=1638 FN=46  (skipped=0)

    Accuracy=75.0%  M-Score=82.1%  Se(Bad)=90.4%  Sp(Good)=73.8%  (evaluated=6726 切片)

    推理耗时 mean=13.88ms  median=13.84ms  p95=14.27ms  min=13.60ms  max=17.75ms  std=0.20ms

  

  [INT8动态]  SQA=heart_quality_quant.tflite

  [load] heart_quality_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

    INT8动态: 100%|█████████████████████████████████████████████████████████████████| 324/324 [02:09<00:00,  2.50file/s]

    TP=434 TN=4605 FP=1640 FN=47  (skipped=0)

    Accuracy=74.9%  M-Score=82.0%  Se(Bad)=90.2%  Sp(Good)=73.7%  (evaluated=6726 切片)

    推理耗时 mean=13.86ms  median=13.83ms  p95=14.16ms  min=13.59ms  max=16.73ms  std=0.15ms

  

  [INT8全整型]  SQA=heart_quality_int8full.tflite

  [load] heart_quality_int8full.tflite  in=int8  out=int8  in_scale=0.324692  out_scale=0.042427

    INT8全整型: 100%|███████████████████████████████████████████████████████████████| 324/324 [01:35<00:00,  3.38file/s]

    TP=434 TN=4621 FP=1624 FN=47  (skipped=0)

    Accuracy=75.2%  M-Score=82.1%  Se(Bad)=90.2%  Sp(Good)=74.0%  (evaluated=6726 切片)

    推理耗时 mean=8.87ms  median=8.85ms  p95=8.94ms  min=8.77ms  max=19.01ms  std=0.13ms

  

========================================================================

FP32 vs INT8动态 vs INT8全整型（SQA 模型）

========================================================================

  Metric               FP32     INT8动态    INT8全整型

  --------------------------------------------------------

  M-Score             82.1%      82.0%      82.1%

  Se(Bad)             90.4%      90.2%      90.2%

  Sp(Good)            73.8%      73.7%      74.0%

  Accuracy            75.0%      74.9%      75.2%

========================================================================

  

  

============================================================

诊断模型评估（解耦，无 SQA 门控，test_split.csv）

  测试录音数：288

============================================================

  

  [FP32]  DIAG=heart_model_fp32.tflite

  [load] heart_model_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

    FP32: 100%|█████████████████████████████████████████████████████████████████████| 288/288 [02:00<00:00,  2.40file/s]

    Accuracy=84.2%  M-Score=87.1%  Se=91.7%  Sp=82.4%  (evaluated=6273 切片, skipped=0 文件)

    推理耗时 mean=13.46ms  median=13.41ms  p95=13.82ms  min=13.14ms  max=14.43ms  std=0.18ms

  

  [INT8动态]  DIAG=heart_model_quant.tflite

  [load] heart_model_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

    INT8动态: 100%|█████████████████████████████████████████████████████████████████| 288/288 [01:58<00:00,  2.44file/s]

    Accuracy=84.1%  M-Score=87.0%  Se=91.7%  Sp=82.3%  (evaluated=6273 切片, skipped=0 文件)

    推理耗时 mean=13.46ms  median=13.42ms  p95=13.82ms  min=13.08ms  max=16.43ms  std=0.19ms

  

  [INT8全整型]  DIAG=heart_model_int8full.tflite

  [load] heart_model_int8full.tflite  in=int8  out=int8  in_scale=0.325044  out_scale=0.069440

    INT8全整型: 100%|███████████████████████████████████████████████████████████████| 288/288 [01:29<00:00,  3.21file/s]

    Accuracy=84.1%  M-Score=87.4%  Se=92.7%  Sp=82.1%  (evaluated=6273 切片, skipped=0 文件)

    推理耗时 mean=8.88ms  median=8.87ms  p95=8.96ms  min=8.80ms  max=9.27ms  std=0.04ms

  

========================================================================

FP32 vs INT8动态 vs INT8全整型（诊断模型，解耦）

========================================================================

  Metric               FP32     INT8动态    INT8全整型

  --------------------------------------------------------

  M-Score             87.1%      87.0%      87.4%

  Sensitivity         91.7%      91.7%      92.7%

  Specificity         82.4%      82.3%      82.1%

  Accuracy            84.2%      84.1%      84.1%

========================================================================

  

  

============================================================

诊断模型评估（耦合：SQA 门控 + 加权平均，test_split.csv）

  测试录音数：288

  SQA_THRESHOLD=0.5（sm[1] 分数，低于此值的窗口被过滤）

============================================================

  

  [FP32]  SQA=heart_quality_fp32.tflite  DIAG=heart_model_fp32.tflite

  [load] heart_quality_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  [load] heart_model_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

    FP32: 100%|█████████████████████████████████████████████████████████████████████| 288/288 [02:22<00:00,  2.02file/s]

    Accuracy=67.8%  M-Score=75.3%  Se=100.0%  Sp=50.5%  (n=143, skipped=145)

    有效窗口/总窗口: 5/21 (平均)

    文件级推理耗时 mean=493.27ms  median=422.41ms  p95=1084.43ms  min=138.25ms  max=1926.20ms  std=282.05ms

  

  [INT8动态]  SQA=heart_quality_quant.tflite  DIAG=heart_model_quant.tflite

  [load] heart_quality_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

  [load] heart_model_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000

    INT8动态: 100%|█████████████████████████████████████████████████████████████████| 288/288 [02:20<00:00,  2.06file/s]

    Accuracy=67.6%  M-Score=75.0%  Se=100.0%  Sp=50.0%  (n=142, skipped=146)

    有效窗口/总窗口: 5/21 (平均)

    文件级推理耗时 mean=485.91ms  median=415.40ms  p95=1063.37ms  min=136.77ms  max=1888.45ms  std=276.87ms

  

  [INT8全整型]  SQA=heart_quality_int8full.tflite  DIAG=heart_model_int8full.tflite

  [load] heart_quality_int8full.tflite  in=int8  out=int8  in_scale=0.324692  out_scale=0.042427

  [load] heart_model_int8full.tflite  in=int8  out=int8  in_scale=0.325044  out_scale=0.069440

    INT8全整型: 100%|███████████████████████████████████████████████████████████████| 288/288 [01:43<00:00,  2.79file/s]

    Accuracy=66.4%  M-Score=73.9%  Se=100.0%  Sp=47.8%  (n=140, skipped=148)

    有效窗口/总窗口: 5/21 (平均)

    文件级推理耗时 mean=357.26ms  median=308.23ms  p95=768.52ms  min=103.62ms  max=1332.85ms  std=197.81ms

  

========================================================================

FP32 vs INT8动态 vs INT8全整型（诊断模型，耦合 SQA 门控）

========================================================================

  Metric               FP32     INT8动态    INT8全整型

  --------------------------------------------------------

  M-Score             75.3%      75.0%      73.9%

  Sensitivity        100.0%     100.0%     100.0%

  Specificity         50.5%      50.0%      47.8%

  Accuracy            67.8%      67.6%      66.4%

========================================================================







(.venv) rasp4b@Rasp4B:~/FypPi $ python evaluate.py --mode both

============================================================
诊断模型评估（耦合：SQA 门控 + 加权平均，test_split.csv）
  测试录音数：288
  SQA_THRESHOLD=0.5（sm[0] 分数，低于此值的窗口被过滤）
============================================================

  [FP32]  SQA=heart_quality_fp32.tflite  DIAG=heart_model_fp32.tflite
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
  [load] heart_quality_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
  [load] heart_model_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
    FP32: 100%|███████████████████████████████████████████████████████████████████████████████████| 288/288 [03:19<00:00,  1.45file/s]
    Accuracy=89.5%  M-Score=87.6%  Se=85.2%  Sp=90.0%  (n=238, skipped=50)
    有效窗口/总窗口: 16/21 (平均)
    文件级推理耗时 mean=690.69ms  median=640.27ms  p95=1321.87ms  min=142.63ms  max=3690.65ms  std=406.62ms

  [INT8动态]  SQA=heart_quality_quant.tflite  DIAG=heart_model_quant.tflite
  [load] heart_quality_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
  [load] heart_model_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
    INT8动态: 100%|███████████████████████████████████████████████████████████████████████████████| 288/288 [03:06<00:00,  1.55file/s]
    Accuracy=89.5%  M-Score=87.6%  Se=85.2%  Sp=90.0%  (n=238, skipped=50)
    有效窗口/总窗口: 16/21 (平均)
    文件级推理耗时 mean=645.52ms  median=614.43ms  p95=1229.13ms  min=140.15ms  max=2035.20ms  std=342.55ms

  [INT8全整型]  SQA=heart_quality_int8full.tflite  DIAG=heart_model_int8full.tflite
  [load] heart_quality_int8full.tflite  in=int8  out=int8  in_scale=0.324692  out_scale=0.042427
  [load] heart_model_int8full.tflite  in=int8  out=int8  in_scale=0.325044  out_scale=0.069440
    INT8全整型: 100%|█████████████████████████████████████████████████████████████████████████████| 288/288 [02:10<00:00,  2.21file/s]
    Accuracy=89.5%  M-Score=87.6%  Se=85.2%  Sp=90.0%  (n=238, skipped=50)
    有效窗口/总窗口: 16/21 (平均)
    文件级推理耗时 mean=452.26ms  median=426.72ms  p95=852.44ms  min=104.48ms  max=1409.42ms  std=237.65ms

========================================================================
FP32 vs INT8动态 vs INT8全整型（诊断模型，耦合 SQA 门控）
========================================================================
  Metric               FP32     INT8动态    INT8全整型
  --------------------------------------------------------
  M-Score             87.6%      87.6%      87.6%
  Sensitivity         85.2%      85.2%      85.2%
  Specificity         90.0%      90.0%      90.0%
  Accuracy            89.5%      89.5%      89.5%
========================================================================

(.venv) rasp4b@Rasp4B:~/FypPi $ python evaluate.py --mode both

============================================================
诊断模型评估（耦合：SQA 门控 + 加权平均，test_split.csv）
  测试录音数：288
  SQA_THRESHOLD=0.65（sm[0] 分数，低于此值的窗口被过滤）
============================================================

  [FP32]  SQA=heart_quality_fp32.tflite  DIAG=heart_model_fp32.tflite
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
  [load] heart_quality_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
  [load] heart_model_fp32.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
    FP32: 100%|███████████████████████████████████████████████████████████████████████████████████| 288/288 [03:13<00:00,  1.49file/s]
    Accuracy=90.2%  M-Score=87.5%  Se=84.0%  Sp=91.0%  (n=235, skipped=53)
    有效窗口/总窗口: 15/21 (平均)
    文件级推理耗时 mean=669.90ms  median=633.13ms  p95=1255.95ms  min=146.16ms  max=3373.53ms  std=385.38ms

  [INT8动态]  SQA=heart_quality_quant.tflite  DIAG=heart_model_quant.tflite
  [load] heart_quality_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
  [load] heart_model_quant.tflite  in=float32  out=float32  in_scale=0.000000  out_scale=0.000000
    INT8动态: 100%|███████████████████████████████████████████████████████████████████████████████| 288/288 [02:59<00:00,  1.60file/s]
    Accuracy=90.2%  M-Score=87.5%  Se=84.0%  Sp=91.0%  (n=235, skipped=53)
    有效窗口/总窗口: 15/21 (平均)
    文件级推理耗时 mean=623.31ms  median=597.82ms  p95=1184.26ms  min=137.74ms  max=2000.82ms  std=332.14ms

  [INT8全整型]  SQA=heart_quality_int8full.tflite  DIAG=heart_model_int8full.tflite
  [load] heart_quality_int8full.tflite  in=int8  out=int8  in_scale=0.324692  out_scale=0.042427
  [load] heart_model_int8full.tflite  in=int8  out=int8  in_scale=0.325044  out_scale=0.069440
    INT8全整型: 100%|█████████████████████████████████████████████████████████████████████████████| 288/288 [02:08<00:00,  2.24file/s]
    Accuracy=89.8%  M-Score=87.2%  Se=84.0%  Sp=90.5%  (n=235, skipped=53)
    有效窗口/总窗口: 15/21 (平均)
    文件级推理耗时 mean=445.60ms  median=424.04ms  p95=845.53ms  min=104.23ms  max=1412.86ms  std=234.80ms

========================================================================
FP32 vs INT8动态 vs INT8全整型（诊断模型，耦合 SQA 门控）
========================================================================
  Metric               FP32     INT8动态    INT8全整型
  --------------------------------------------------------
  M-Score             87.5%      87.5%      87.2%
  Sensitivity         84.0%      84.0%      84.0%
  Specificity         91.0%      91.0%      90.5%
  Accuracy            90.2%      90.2%      89.8%
========================================================================