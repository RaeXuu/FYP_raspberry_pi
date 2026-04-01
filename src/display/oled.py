import threading
import time
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306, sh1106
from luma.core.render import canvas


class OLEDDisplay:
    def __init__(self, port=1, address=0x3C):
        serial = i2c(port=port, address=address)
        self.device = ssd1306(serial)
        self._lock = threading.Lock()

    def _draw(self, fn):
        with self._lock:
            with canvas(self.device) as draw:
                fn(draw)

    # --------------------------------------------------
    # 1. 启动页
    # --------------------------------------------------
    def show_boot(self):
        def fn(draw):
            draw.text((16, 0),  "Heart Sound",  fill="white")
            draw.text((24, 10), "Diagnosis",    fill="white")
            draw.text((44, 22), "v1.0",         fill="white")
        self._draw(fn)

    # --------------------------------------------------
    # 2. 连接中（progress: 0.0–1.0）
    # --------------------------------------------------
    def show_connecting(self, progress=0.0):
        bar_w = 120
        filled = int(bar_w * progress)

        def fn(draw):
            draw.text((0, 0),  "Connecting...", fill="white")
            draw.rectangle([0, 14, bar_w, 22], outline="white")
            if filled > 0:
                draw.rectangle([0, 14, filled, 22], fill="white")
        self._draw(fn)

    # --------------------------------------------------
    # 3. 运行中（含上次结果）
    #    normal_pct / abnormal_pct: 0–100 float
    #    chunk_idx: 当前块编号
    #    last_label: "NORMAL" / "ABNORMAL" / None
    # --------------------------------------------------
    def show_running(self, normal_pct=None, abnormal_pct=None,
                     chunk_idx=0, last_label=None):
        if normal_pct is None:
            n_str = "Normal:    --"
            a_str = "Abnormal:  --"
        else:
            n_str = f"Normal:  {normal_pct:5.1f}%"
            a_str = f"Abnormal:{abnormal_pct:5.1f}%"

        last_str = f"Last: {last_label}" if last_label else "Last: --"

        def fn(draw):
            draw.text((0, 0),  f"Analyzing #{chunk_idx:03d}", fill="white")
            draw.text((0, 10), n_str,                          fill="white")
            draw.text((0, 20), a_str,                          fill="white")
            draw.text((0, 30) if last_label else (0, 30),
                      last_str, fill="white")
        self._draw(fn)

    # --------------------------------------------------
    # 4. 错误页
    # --------------------------------------------------
    def show_error(self, msg="BLE Lost!"):
        def fn(draw):
            draw.text((0, 8),  msg,   fill="white")
            draw.text((0, 20), "Retry: press btn", fill="white")
        self._draw(fn)

    def show_standby(self):
        def fn(draw):
            draw.text((10, 2),  "Heart Sound", fill="white")
            draw.text((4, 14),  "Diagnosis  v1.0", fill="white")
            draw.text((4, 26),  "Press to start", fill="white")
        self._draw(fn)

    def show_text(self, msg):
        def fn(draw):
            draw.text((0, 16), msg, fill="white")
        self._draw(fn)

    def cleanup(self):
        self.device.cleanup()


class SysInfoDisplay:
    """第二块屏（128x32 ssd1306，i2c-4），显示系统状态。"""
    def __init__(self, port=4, address=0x3C):
        serial = i2c(port=port, address=address)
        self.device = ssd1306(serial, width=128, height=32, rotate=0)
        self._lock = threading.Lock()

    def show(self, cpu_pct, mem_used_mb, mem_total_mb, temp_c):
        def fn(draw):
            draw.text((0, 0),  f"CPU: {cpu_pct:5.1f}%",              fill="white")
            draw.text((0, 9),  f"Mem: {mem_used_mb:.0f}/{mem_total_mb:.0f}MB", fill="white")
            draw.text((0, 18), f"Tmp: {temp_c:.0f}C",                fill="white")
        with self._lock:
            with canvas(self.device) as draw:
                fn(draw)

    def cleanup(self):
        self.device.cleanup()


if __name__ == "__main__":
    import time
    import random

    oled = OLEDDisplay()

    print("1. 启动页")
    oled.show_boot()
    time.sleep(3)

    print("2. 待机页")
    oled.show_standby()
    time.sleep(3)

    print("3. 连接中（进度动画）")
    for i in range(11):
        oled.show_connecting(i / 10)
        time.sleep(0.3)

    print("4. 运行中（模拟窗口更新）")
    last = None
    for chunk in range(1, 4):
        for win in range(1, 10):
            n = random.uniform(40, 90)
            oled.show_running(n, 100 - n, chunk, last)
            time.sleep(0.3)
        last = "NORMAL" if n > 50 else "ABNORMAL"

    print("5. 错误页")
    oled.show_error("BLE Lost!")
    time.sleep(3)

    oled.cleanup()
    print("完成")
