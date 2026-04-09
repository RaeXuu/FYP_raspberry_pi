  # 延迟测试（用几条真实 WAV，比如随便挑几条 test 集里的）
  .venv/bin/python benchmark.py --wav data/raw/DataSet2/training-a/a0002.wav

  # 准确率评估（诊断 + SQA，FP32 vs INT8）
  .venv/bin/python evaluate.py --mode both


**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py --wav data/raw/DataSet2/training-a/a0002.wav

使用 WAV 文件：data/raw/DataSet2/training-a/a0002.wav

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  

============================================================

模型文件大小

============================================================

  SQA  FP32    302.8 KB

  SQA  INT8    144.7 KB

  Diag FP32    302.8 KB

  Diag INT8    144.7 KB

  

============================================================

各阶段延迟（median of 100 runs，单个 2s 窗口）

============================================================

  Bandpass filter              median=2.23ms  mean=2.49  min=2.15  max=25.71  (n=100)

  Log-Mel spectrogram          median=4.80ms  mean=81.36  min=4.72  max=7658.71  (n=100)

  SQA model     FP32           median=13.49ms  mean=13.75  min=13.31  max=30.19  (n=100)

  SQA model     INT8           median=13.48ms  mean=13.58  min=13.30  max=20.50  (n=100)

  Diag model    FP32           median=13.51ms  mean=13.56  min=13.33  max=14.29  (n=100)

  Diag model    INT8           median=13.44ms  mean=13.48  min=13.29  max=14.06  (n=100)

  

============================================================

FP32 vs INT8 对比（Table 6.1）

============================================================

  Stage                         FP32 (ms)  INT8 (ms)    Speedup

  ----------------------------------------------------------

  Bandpass filter                       —          —          —

  Log-Mel spectrogram                   —          —          —

  SQA model                        13.49ms     13.48ms      1.00x

  Diag model                       13.51ms     13.44ms      1.01x

  Total per segment                34.03ms     33.94ms

  

  实时性约束：< 2000ms/segment

  FP32 总延迟：34.0ms  ✓

  INT8 总延迟：33.9ms  ✓

  

============================================================

系统资源占用（Table 6.2）

============================================================

  Peak CPU utilisation   1.0%

  Memory (RSS)           249.2 MB

  CPU temperature        44.8 °C

============================================================


**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py --wav data/raw/DataSet2/training-b/b0001.wav

使用 WAV 文件：data/raw/DataSet2/training-b/b0001.wav

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  

============================================================

模型文件大小

============================================================

  SQA  FP32    302.8 KB

  SQA  INT8    144.7 KB

  Diag FP32    302.8 KB

  Diag INT8    144.7 KB

  

============================================================

各阶段延迟（median of 100 runs，单个 2s 窗口）

============================================================

  Bandpass filter              median=2.31ms  mean=2.33  min=2.19  max=3.13  (n=100)

  Log-Mel spectrogram          median=4.70ms  mean=30.08  min=4.63  max=2540.30  (n=100)

  SQA model     FP32           median=13.46ms  mean=13.53  min=13.34  max=14.66  (n=100)

  SQA model     INT8           median=13.48ms  mean=13.51  min=13.33  max=14.17  (n=100)

  Diag model    FP32           median=13.42ms  mean=13.48  min=13.26  max=14.26  (n=100)

  Diag model    INT8           median=13.44ms  mean=13.48  min=13.28  max=14.23  (n=100)

  

============================================================

FP32 vs INT8 对比（Table 6.1）

============================================================

  Stage                         FP32 (ms)  INT8 (ms)    Speedup

  ----------------------------------------------------------

  Bandpass filter                       —          —          —

  Log-Mel spectrogram                   —          —          —

  SQA model                        13.46ms     13.48ms      1.00x

  Diag model                       13.42ms     13.44ms      1.00x

  Total per segment                33.91ms     33.94ms

  

  实时性约束：< 2000ms/segment

  FP32 总延迟：33.9ms  ✓

  INT8 总延迟：33.9ms  ✓

  

============================================================

系统资源占用（Table 6.2）

============================================================

  Peak CPU utilisation   1.7%

  Memory (RSS)           249.9 MB

  CPU temperature        42.8 °C

============================================================


**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py --wav data/raw/DataSet2/training-c/c0004.wav

使用 WAV 文件：data/raw/DataSet2/training-c/c0004.wav

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  

============================================================

模型文件大小

============================================================

  SQA  FP32    302.8 KB

  SQA  INT8    144.7 KB

  Diag FP32    302.8 KB

  Diag INT8    144.7 KB

  

============================================================

各阶段延迟（median of 100 runs，单个 2s 窗口）

============================================================

  Bandpass filter              median=2.23ms  mean=2.25  min=2.15  max=3.19  (n=100)

  Log-Mel spectrogram          median=4.73ms  mean=30.16  min=4.65  max=2543.97  (n=100)

  SQA model     FP32           median=13.62ms  mean=13.65  min=13.47  max=14.26  (n=100)

  SQA model     INT8           median=13.49ms  mean=13.53  min=13.33  max=14.15  (n=100)

  Diag model    FP32           median=13.43ms  mean=13.50  min=13.30  max=14.27  (n=100)

  Diag model    INT8           median=13.41ms  mean=13.44  min=13.25  max=13.92  (n=100)

  

============================================================

FP32 vs INT8 对比（Table 6.1）

============================================================

  Stage                         FP32 (ms)  INT8 (ms)    Speedup

  ----------------------------------------------------------

  Bandpass filter                       —          —          —

  Log-Mel spectrogram                   —          —          —

  SQA model                        13.62ms     13.49ms      1.01x

  Diag model                       13.43ms     13.41ms      1.00x

  Total per segment                34.01ms     33.86ms

  

  实时性约束：< 2000ms/segment

  FP32 总延迟：34.0ms  ✓

  INT8 总延迟：33.9ms  ✓

  

============================================================

系统资源占用（Table 6.2）

============================================================

  Peak CPU utilisation   1.7%

  Memory (RSS)           250.3 MB

  CPU temperature        43.8 °C

============================================================


**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py --wav data/raw/DataSet2/training-d/d0005.wav

使用 WAV 文件：data/raw/DataSet2/training-d/d0005.wav

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  

============================================================

模型文件大小

============================================================

  SQA  FP32    302.8 KB

  SQA  INT8    144.7 KB

  Diag FP32    302.8 KB

  Diag INT8    144.7 KB

  

============================================================

各阶段延迟（median of 100 runs，单个 2s 窗口）

============================================================

  Bandpass filter              median=2.22ms  mean=2.24  min=2.16  max=3.01  (n=100)

  Log-Mel spectrogram          median=4.76ms  mean=30.10  min=4.68  max=2535.82  (n=100)

  SQA model     FP32           median=13.44ms  mean=13.47  min=13.28  max=14.03  (n=100)

  SQA model     INT8           median=13.43ms  mean=13.48  min=13.30  max=14.09  (n=100)

  Diag model    FP32           median=13.40ms  mean=13.45  min=13.26  max=14.25  (n=100)

  Diag model    INT8           median=13.41ms  mean=13.45  min=13.23  max=13.96  (n=100)

  

============================================================

FP32 vs INT8 对比（Table 6.1）

============================================================

  Stage                         FP32 (ms)  INT8 (ms)    Speedup

  ----------------------------------------------------------

  Bandpass filter                       —          —          —

  Log-Mel spectrogram                   —          —          —

  SQA model                        13.44ms     13.43ms      1.00x

  Diag model                       13.40ms     13.41ms      1.00x

  Total per segment                33.82ms     33.82ms

  

  实时性约束：< 2000ms/segment

  FP32 总延迟：33.8ms  ✓

  INT8 总延迟：33.8ms  ✓

  

============================================================

系统资源占用（Table 6.2）

============================================================

  Peak CPU utilisation   0.5%

  Memory (RSS)           250.0 MB

  CPU temperature        43.8 °C

============================================================


**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py --wav data/raw/DataSet2/training-e/e00011.wav

使用 WAV 文件：data/raw/DataSet2/training-e/e00011.wav

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  

============================================================

模型文件大小

============================================================

  SQA  FP32    302.8 KB

  SQA  INT8    144.7 KB

  Diag FP32    302.8 KB

  Diag INT8    144.7 KB

  

============================================================

各阶段延迟（median of 100 runs，单个 2s 窗口）

============================================================

  Bandpass filter              median=2.26ms  mean=2.29  min=2.15  max=3.14  (n=100)

  Log-Mel spectrogram          median=4.68ms  mean=30.02  min=4.61  max=2536.04  (n=100)

  SQA model     FP32           median=13.62ms  mean=13.66  min=13.44  max=14.51  (n=100)

  SQA model     INT8           median=13.57ms  mean=13.60  min=13.36  max=14.09  (n=100)

  Diag model    FP32           median=13.48ms  mean=13.53  min=13.31  max=14.27  (n=100)

  Diag model    INT8           median=13.51ms  mean=13.55  min=13.34  max=14.12  (n=100)

  

============================================================

FP32 vs INT8 对比（Table 6.1）

============================================================

  Stage                         FP32 (ms)  INT8 (ms)    Speedup

  ----------------------------------------------------------

  Bandpass filter                       —          —          —

  Log-Mel spectrogram                   —          —          —

  SQA model                        13.62ms     13.57ms      1.00x

  Diag model                       13.48ms     13.51ms      1.00x

  Total per segment                34.05ms     34.03ms

  

  实时性约束：< 2000ms/segment

  FP32 总延迟：34.1ms  ✓

  INT8 总延迟：34.0ms  ✓

  

============================================================

系统资源占用（Table 6.2）

============================================================

  Peak CPU utilisation   0.8%

  Memory (RSS)           250.0 MB

  CPU temperature        43.8 °C

============================================================


**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python benchmark.py --wav data/raw/DataSet2/training-f/f0001.wav

使用 WAV 文件：data/raw/DataSet2/training-f/f0001.wav

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

  

============================================================

模型文件大小

============================================================

  SQA  FP32    302.8 KB

  SQA  INT8    144.7 KB

  Diag FP32    302.8 KB

  Diag INT8    144.7 KB

  

============================================================

各阶段延迟（median of 100 runs，单个 2s 窗口）

============================================================

  Bandpass filter              median=2.20ms  mean=2.21  min=2.13  max=3.14  (n=100)

  Log-Mel spectrogram          median=4.70ms  mean=30.11  min=4.63  max=2542.04  (n=100)

  SQA model     FP32           median=13.42ms  mean=13.46  min=13.24  max=14.17  (n=100)

  SQA model     INT8           median=13.31ms  mean=13.36  min=13.11  max=14.03  (n=100)

  Diag model    FP32           median=13.39ms  mean=13.44  min=13.24  max=13.99  (n=100)

  Diag model    INT8           median=13.39ms  mean=13.43  min=13.24  max=14.05  (n=100)

  

============================================================

FP32 vs INT8 对比（Table 6.1）

============================================================

  Stage                         FP32 (ms)  INT8 (ms)    Speedup

  ----------------------------------------------------------

  Bandpass filter                       —          —          —

  Log-Mel spectrogram                   —          —          —

  SQA model                        13.42ms     13.31ms      1.01x

  Diag model                       13.39ms     13.39ms      1.00x

  Total per segment                33.71ms     33.60ms

  

  实时性约束：< 2000ms/segment

  FP32 总延迟：33.7ms  ✓

  INT8 总延迟：33.6ms  ✓

  

============================================================

系统资源占用（Table 6.2）

============================================================

  Peak CPU utilisation   2.0%

  Memory (RSS)           250.1 MB

  CPU temperature        44.8 °C

============================================================


**rasp4b@Rasp4B**:**~/FypPi $** .venv/bin/python evaluate.py --mode both

  

============================================================

诊断模型评估（test_split.csv，test_split.csv）

  测试录音数：288

============================================================

  

  [FP32]  SQA=heart_quality_fp32.tflite  DIAG=heart_model_fp32.tflite

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

    FP32: 100%|█████████████████████████████████████████████████████████████████████| 288/288 [02:22<00:00,  2.02file/s]

    Accuracy=65.4%  M-Score=68.9%  Se=97.9%  Sp=40.0%  (evaluated=107, skipped=181)

    推理耗时 mean=493ms  min=139ms  max=4622ms

  

  [INT8]  SQA=heart_quality_quant.tflite  DIAG=heart_model_quant.tflite

    INT8: 100%|█████████████████████████████████████████████████████████████████████| 288/288 [02:14<00:00,  2.13file/s]

    Accuracy=65.4%  M-Score=68.9%  Se=97.9%  Sp=40.0%  (evaluated=107, skipped=181)

    推理耗时 mean=468ms  min=131ms  max=1920ms

  

============================================================

FP32 vs INT8 对比（诊断模型，Table 5.5 / Table 6.3）

============================================================

  Metric               FP32       INT8     Change

  --------------------------------------------

  M-Score             68.9%      68.9%      +0.0%

  Sensitivity         97.9%      97.9%      +0.0%

  Specificity         40.0%      40.0%      +0.0%

  Accuracy            65.4%      65.4%      +0.0%

============================================================

  

  

============================================================

SQA 模型评估（test_split_sqa.csv，test_split_sqa.csv）

  测试录音数：324

============================================================

  

  [FP32]  SQA=heart_quality_fp32.tflite

    FP32: 100%|█████████████████████████████████████████████████████████████████████| 324/324 [02:12<00:00,  2.44file/s]

    Accuracy=80.6%  M-Score=82.3%  Se(Bad)=84.4%  Sp(Good)=80.1%  (evaluated=324, skipped=0)

  

  [INT8]  SQA=heart_quality_quant.tflite

    INT8: 100%|█████████████████████████████████████████████████████████████████████| 324/324 [02:10<00:00,  2.48file/s]

    Accuracy=80.6%  M-Score=82.3%  Se(Bad)=84.4%  Sp(Good)=80.1%  (evaluated=324, skipped=0)

  

============================================================

FP32 vs INT8 对比（SQA 模型）

============================================================

  Metric               FP32       INT8     Change

  --------------------------------------------

  M-Score             82.3%      82.3%      +0.0%

  Se(Bad)             84.4%      84.4%      +0.0%

  Sp(Good)            80.1%      80.1%      +0.0%

  Accuracy            80.6%      80.6%      +0.0%

============================================================