import asyncio
import re
import time
import serial

SERIAL_PORT = '/dev/ttyS0'
BAUD_RATE   = 115200
ATE_INTERVAL = 60   # 每 60 秒查询一次运行时长

# Li-ion 电压 → 电量百分比（线性近似，3.0V=0%，4.2V=100%）
_V_MIN = 3.0
_V_MAX = 4.2

# ATE 响应格式：DDD:HH:MM:SS
_ATE_RE = re.compile(r'^(\d+):(\d{2}):(\d{2}):(\d{2})\s*$')


def _voltage_to_pct(v: float) -> float:
    pct = (v - _V_MIN) / (_V_MAX - _V_MIN) * 100.0
    return max(0.0, min(100.0, pct))


def _fmt_uptime(days: int, hours: int, minutes: int, seconds: int) -> str:
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class PowerReader:
    def __init__(self, port: str = SERIAL_PORT):
        self._port      = port
        self._battery_v: float | None = None
        self._dcinput_v: float | None = None
        self._uptime:    str | None   = None
        self._task: asyncio.Task | None = None
        self._ser: serial.Serial | None = None

    # ---- 对外读接口 ----

    def bat_pct(self) -> float | None:
        return _voltage_to_pct(self._battery_v) if self._battery_v is not None else None

    def battery_v(self) -> float | None:
        return self._battery_v

    def dcinput_v(self) -> float | None:
        return self._dcinput_v

    def uptime(self) -> str | None:
        return self._uptime

    # ---- 内部 ----

    def _parse(self, raw: bytes) -> None:
        try:
            line = raw.decode('utf-8', errors='ignore').strip()
        except Exception:
            return

        # 尝试解析 ATE 运行时长（含中文，优先判断）
        m = _ATE_RE.search(line)
        if m:
            self._uptime = _fmt_uptime(int(m.group(1)), int(m.group(2)),
                                        int(m.group(3)), int(m.group(4)))
            return

        # 解析电压上报 KEY:X.XXV
        if ':' not in line:
            return
        key, _, val = line.partition(':')
        try:
            v = float(val.rstrip('V'))
        except ValueError:
            return
        key = key.strip()
        if key == 'BATTERY':
            self._battery_v = v
        elif key == 'DCINPUT':
            self._dcinput_v = v

    def _readline(self) -> bytes:
        return self._ser.readline()

    async def _loop(self) -> None:
        try:
            self._ser = serial.Serial(self._port, BAUD_RATE, timeout=0.5)
            loop = asyncio.get_event_loop()
            last_ate = time.monotonic() - ATE_INTERVAL  # 启动后立即查一次
            while True:
                now = time.monotonic()
                if now - last_ate >= ATE_INTERVAL:
                    self._ser.write(b'ATE\r\n')
                    last_ate = now

                raw = await loop.run_in_executor(None, self._readline)
                if raw:
                    self._parse(raw)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[PowerReader] 串口异常: {e}")
        finally:
            if self._ser and self._ser.is_open:
                self._ser.close()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
