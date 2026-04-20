import asyncio
import RPi.GPIO as GPIO

PIN_R = 17
PIN_G = 22
PIN_B = 10
BRIGHTNESS = 0.3


class RGBLed:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        for pin in (PIN_R, PIN_G, PIN_B):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        self._r = GPIO.PWM(PIN_R, 100)
        self._g = GPIO.PWM(PIN_G, 100)
        self._b = GPIO.PWM(PIN_B, 100)
        self._r.start(0)
        self._g.start(0)
        self._b.start(0)
        self._task = None

    def _set(self, r, g, b):
        self._r.ChangeDutyCycle(r * BRIGHTNESS)
        self._g.ChangeDutyCycle(g * BRIGHTNESS)
        self._b.ChangeDutyCycle(b * BRIGHTNESS)

    def _cancel(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _blink(self, r, g, b, hz=1.0):
        interval = 1.0 / hz / 2
        try:
            while True:
                self._set(r, g, b)
                await asyncio.sleep(interval)
                self._set(0, 0, 0)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def standby(self):
        self._cancel()
        self._set(0, 0, 100)

    def connecting(self):
        self._cancel()
        self._task = asyncio.get_event_loop().create_task(
            self._blink(0, 0, 100, hz=1.0)
        )

    async def _breathe(self, r, g, b, period=2.0):
        import math
        step = 0.05
        try:
            while True:
                t = 0.0
                while t < period:
                    brightness = (1 - math.cos(2 * math.pi * t / period)) / 2
                    self._set(r * brightness, g * brightness, b * brightness)
                    await asyncio.sleep(step)
                    t += step
        except asyncio.CancelledError:
            pass

    def running(self):
        self._cancel()
        self._task = asyncio.get_event_loop().create_task(
            self._breathe(0, 0, 100, period=2.0)
        )

    def normal(self):
        self._cancel()
        self._set(0, 100, 0)

    def abnormal(self):
        self._cancel()
        self._set(100, 0, 0)

    def error(self):
        self._cancel()
        self._task = asyncio.get_event_loop().create_task(
            self._blink(100, 0, 0, hz=4.0)
        )

    def off(self):
        self._cancel()
        self._set(0, 0, 0)

    def cleanup(self):
        self._cancel()
        self._set(0, 0, 0)
        self._r.stop()
        self._g.stop()
        self._b.stop()
        GPIO.cleanup([PIN_R, PIN_G, PIN_B])
