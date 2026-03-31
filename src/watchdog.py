"""
软件看门狗：监控 /tmp/heartbeat.ts，超时则重启 heartbeat 服务。

独立运行，不依赖主程序。由 watchdog.service 管理。
"""
import os
import subprocess
import time

HEARTBEAT_PATH = "/tmp/heartbeat.ts"
CHECK_INTERVAL = 30   # 每隔多久检查一次（秒）
TIMEOUT        = 90   # 超过多久没更新视为主程序挂死（秒）
SERVICE        = "heartbeat"


def check_and_restart():
    now = time.time()

    if not os.path.exists(HEARTBEAT_PATH):
        print(f"[看门狗] 心跳文件不存在，重启 {SERVICE}...")
        _restart()
        return

    try:
        last = float(open(HEARTBEAT_PATH).read().strip())
    except (ValueError, OSError):
        print(f"[看门狗] 心跳文件读取失败，重启 {SERVICE}...")
        _restart()
        return

    elapsed = now - last
    if elapsed > TIMEOUT:
        print(f"[看门狗] 心跳超时 {elapsed:.0f}s > {TIMEOUT}s，重启 {SERVICE}...")
        _restart()
    else:
        print(f"[看门狗] 心跳正常，距上次更新 {elapsed:.0f}s")


def _restart():
    subprocess.run(["systemctl", "restart", SERVICE], check=False)


if __name__ == "__main__":
    print(f"[看门狗] 启动，每 {CHECK_INTERVAL}s 检查一次，超时阈值 {TIMEOUT}s")
    while True:
        time.sleep(CHECK_INTERVAL)
        check_and_restart()
