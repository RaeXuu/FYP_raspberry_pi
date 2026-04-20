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
        self._blink_stop   = threading.Event()
        self._blink_thread = None
        self._conn_stop    = threading.Event()
        self._conn_thread  = None

    def _draw(self, fn):
        with self._lock:
            with canvas(self.device) as draw:
                fn(draw)


    # --------------------------------------------------
    # 2. 连接中
    # --------------------------------------------------
    def show_connecting(self, progress=0.0, timeout_left=15):
        bar_w = 120
        filled = int(bar_w * progress)

        def fn(draw):
            draw.text((0,  4), "Connecting ESP32...",         fill="white")
            draw.rectangle([0, 18, bar_w, 26], outline="white")
            if filled > 0:
                draw.rectangle([0, 18, filled, 26], fill="white")
            draw.text((0, 36), f"Timeout: {timeout_left:2d}s", fill="white")
        self._draw(fn)

    def start_connecting_countdown(self, timeout=15):
        self._conn_stop.clear()
        self._conn_thread = threading.Thread(
            target=self._conn_loop, args=(timeout,), daemon=True)
        self._conn_thread.start()

    def stop_connecting_countdown(self):
        self._conn_stop.set()
        if self._conn_thread:
            self._conn_thread.join(timeout=2)
            self._conn_thread = None

    def _conn_loop(self, timeout):
        start = time.time()
        while not self._conn_stop.is_set():
            elapsed  = time.time() - start
            progress = min(1.0, elapsed / timeout)
            remaining = max(0, timeout - int(elapsed))
            self.show_connecting(progress, remaining)
            self._conn_stop.wait(timeout=1.0)

    # --------------------------------------------------
    # 3. 运行中（含上次结果）
    #    normal_pct / abnormal_pct: 0–100 float
    #    chunk_idx: 当前块编号
    #    last_label: "NORMAL" / "ABNORMAL" / None
    # --------------------------------------------------
    def show_running(self, normal_pct=None, chunk_idx=0,
                     last_label=None, last_chunk_idx=None, last_prob=None,
                     win_idx=0, total_win=0, heart_on=True):
        n_str    = f"Normal: {normal_pct:5.1f}%" if normal_pct is not None else "Normal:    --"
        win_str  = f"Win: {win_idx:02d}/{total_win:02d}" if total_win > 0 else "Win: --/--"

        if last_label and last_chunk_idx is not None:
            last_hdr  = f"Last: #{last_chunk_idx:03d}"
            last_body = f"{last_label}  {last_prob*100:4.1f}%" if last_prob is not None else last_label
        else:
            last_hdr  = "Last: --"
            last_body = ""

        def fn(draw):
            if heart_on:
                OLEDDisplay._draw_heart(draw, 0, 3)
            draw.text((9,  0), f"> Analyzing #{chunk_idx:03d}", fill="white")
            draw.text((9, 10), win_str,                          fill="white")
            draw.text((9, 20), n_str,                            fill="white")
            draw.line([(0, 35), (127, 35)],                      fill="white")
            OLEDDisplay._draw_heart(draw, 0, 42)
            draw.text((9, 38), last_hdr,                         fill="white")
            draw.text((9, 48), last_body,                        fill="white")
        self._draw(fn)

    # --------------------------------------------------
    # 4. 错误页
    # --------------------------------------------------
    def show_error(self, msg="BLE Lost!"):
        def fn(draw):
            draw.text((0, 8),  msg,   fill="white")
            draw.text((0, 20), "Retry: press btn", fill="white")
        self._draw(fn)

    @staticmethod
    def _draw_heart(draw, x, y):
        """在 (x, y) 处绘制 7×6 像素爱心。"""
        pixels = [
            (x+1,y+0),(x+2,y+0),(x+4,y+0),(x+5,y+0),
            (x+0,y+1),(x+1,y+1),(x+2,y+1),(x+3,y+1),(x+4,y+1),(x+5,y+1),(x+6,y+1),
            (x+0,y+2),(x+1,y+2),(x+2,y+2),(x+3,y+2),(x+4,y+2),(x+5,y+2),(x+6,y+2),
            (x+1,y+3),(x+2,y+3),(x+3,y+3),(x+4,y+3),(x+5,y+3),
            (x+2,y+4),(x+3,y+4),(x+4,y+4),
            (x+3,y+5),
        ]
        draw.point(pixels, fill="white")

    def show_standby(self, heart_on=True):
        def fn(draw):
            draw.rectangle([0, 0, 127, 50], outline="white")
            draw.text(( 4,  1), "Heartsound Diagnosis", fill="white")
            draw.line([(1, 13), (126, 13)],             fill="white")
            draw.text(( 4, 15), "NUSRI  Xu Ruijing",   fill="white")
            draw.text(( 4, 26), "NUSRI  Wang Yulin",   fill="white")
            draw.text(( 4, 37), "Advisor: Prof. Heng", fill="white")
            if heart_on:
                OLEDDisplay._draw_heart(draw, 29, 56)
            draw.text((40, 53), "Press to start ...",   fill="white")
        self._draw(fn)

    def start_standby_blink(self):
        """启动爱心闪烁线程（每秒切换一次）。"""
        self._blink_stop.clear()
        self._blink_thread = threading.Thread(target=self._blink_loop, daemon=True)
        self._blink_thread.start()

    def stop_standby_blink(self):
        """停止爱心闪烁线程。"""
        self._blink_stop.set()
        if self._blink_thread:
            self._blink_thread.join(timeout=2)
            self._blink_thread = None

    def _blink_loop(self):
        heart_on = True
        while not self._blink_stop.is_set():
            self.show_standby(heart_on=heart_on)
            heart_on = not heart_on
            self._blink_stop.wait(timeout=1.0)

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

    def show(self, cpu_pct, mem_used_mb, mem_total_mb, temp_c, wifi_on=True):
        def fn(draw):
            draw.text((0, 0),  f"CPU: {cpu_pct:5.1f}%",                       fill="white")
            draw.text((0, 9),  f"Mem: {mem_used_mb:.0f}/{mem_total_mb:.0f}MB", fill="white")
            draw.text((0, 18), f"Tmp: {temp_c:.0f}C",                          fill="white")
            if wifi_on:
                SysInfoDisplay._draw_wifi(draw, 115, 0)
        with self._lock:
            with canvas(self.device) as draw:
                fn(draw)

    @staticmethod
    def _draw_wifi(draw, x, y):
        """12×8 像素 WiFi 图标。"""
        draw.point([
            # outer arc
            (x+2,y+0),(x+3,y+0),(x+4,y+0),(x+5,y+0),(x+6,y+0),(x+7,y+0),(x+8,y+0),(x+9,y+0),
            (x+1,y+1),(x+10,y+1),
            (x+0,y+2),(x+11,y+2),
            # middle arc
            (x+3,y+3),(x+4,y+3),(x+5,y+3),(x+6,y+3),(x+7,y+3),(x+8,y+3),
            (x+2,y+4),(x+9,y+4),
            # inner arc
            (x+4,y+5),(x+5,y+5),(x+6,y+5),(x+7,y+5),
            # dot (2×2)
            (x+5,y+6),(x+6,y+6),
            (x+5,y+7),(x+6,y+7),
        ], fill="white")

    def show_power(self, bat_pct=None, power_w=None, voltage_v=None, wifi_on=True):
        bat_str = f"Bat: {bat_pct:5.1f}%" if bat_pct  is not None else "Bat:    --%"
        pwr_str = f"Pwr: {power_w:5.2f}W" if power_w  is not None else "Pwr:   -- W"
        vol_str = f"Vol: {voltage_v:4.2f}V" if voltage_v is not None else "Vol:   -- V"

        def fn(draw):
            draw.text((0, 0),  bat_str, fill="white")
            draw.text((0, 11), pwr_str, fill="white")
            draw.text((0, 22), vol_str, fill="white")
            if wifi_on:
                SysInfoDisplay._draw_wifi(draw, 115, 0)
        with self._lock:
            with canvas(self.device) as draw:
                fn(draw)

    def cleanup(self):
        self.device.cleanup()


if __name__ == "__main__":
    import time
    import random

    oled = OLEDDisplay()

    print("1. 待机页（爱心闪烁 5s）")
    oled.start_standby_blink()
    time.sleep(5)
    oled.stop_standby_blink()

    print("2. 连接中（倒计时 5s 演示）")
    oled.start_connecting_countdown(timeout=5)
    time.sleep(5)
    oled.stop_connecting_countdown()

    print("4. 运行中（模拟窗口更新）")
    last_label = None
    last_cidx  = None
    last_prob  = None
    for chunk in range(1, 4):
        for win in range(1, 10):
            n = random.uniform(40, 90)
            oled.show_running(
                normal_pct=n, chunk_idx=chunk,
                last_label=last_label, last_chunk_idx=last_cidx, last_prob=last_prob,
                win_idx=win, total_win=9, heart_on=(win % 2 == 0)
            )
            time.sleep(0.3)
        last_label = "NORMAL" if n > 50 else "ABNORMAL"
        last_cidx  = chunk
        last_prob  = n / 100

    print("5. 错误页")
    oled.show_error("BLE Lost!")
    time.sleep(3)

    oled.cleanup()
    print("完成")
