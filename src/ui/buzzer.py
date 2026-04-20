import asyncio
import RPi.GPIO as GPIO

PIN_BUZZER = 25


class Buzzer:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_BUZZER, GPIO.OUT, initial=GPIO.HIGH)

    async def _beep(self, times, on_ms=100, off_ms=200):
        for i in range(times):
            GPIO.output(PIN_BUZZER, GPIO.LOW)
            await asyncio.sleep(on_ms / 1000)
            GPIO.output(PIN_BUZZER, GPIO.HIGH)
            if i < times - 1:
                await asyncio.sleep(off_ms / 1000)

    def normal(self):
        asyncio.get_event_loop().create_task(self._beep(1))

    def abnormal(self):
        asyncio.get_event_loop().create_task(self._beep(3))

    def cleanup(self):
        GPIO.output(PIN_BUZZER, GPIO.HIGH)
        GPIO.cleanup([PIN_BUZZER])
