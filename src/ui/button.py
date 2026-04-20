import asyncio
import time
import RPi.GPIO as GPIO

BUTTON_PIN    = 15
DEBOUNCE_MS   = 20
LONG_PRESS_S  = 3.0


class Button:
    def __init__(self, pin=BUTTON_PIN):
        self._pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self._short_press_cb = None
        self._long_press_cb  = None
        self._task = None

    def on_short_press(self, cb):
        self._short_press_cb = cb

    def on_long_press(self, cb):
        self._long_press_cb = cb

    async def _call(self, cb):
        if cb is None:
            return
        if asyncio.iscoroutinefunction(cb):
            await cb()
        else:
            cb()

    async def _monitor(self):
        pressed_at = None

        try:
            while True:
                level = GPIO.input(self._pin)

                if level == GPIO.LOW and pressed_at is None:
                    await asyncio.sleep(DEBOUNCE_MS / 1000)
                    if GPIO.input(self._pin) == GPIO.LOW:
                        pressed_at = time.monotonic()

                elif level == GPIO.HIGH and pressed_at is not None:
                    await asyncio.sleep(DEBOUNCE_MS / 1000)
                    if GPIO.input(self._pin) == GPIO.HIGH:
                        duration = time.monotonic() - pressed_at
                        pressed_at = None
                        if duration >= LONG_PRESS_S:
                            await self._call(self._long_press_cb)
                        else:
                            await self._call(self._short_press_cb)

                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Button:{self._pin}] 监听异常: {e}")

    def start(self):
        self._task = asyncio.create_task(self._monitor())

    def stop(self):
        if self._task:
            self._task.cancel()
        GPIO.cleanup(self._pin)


if __name__ == "__main__":
    import asyncio

    async def test():
        btn = Button()
        btn.on_short_press(lambda: print("短按"))
        btn.on_long_press(lambda: print("长按"))
        btn.start()
        print("等待按键（30s）...")
        await asyncio.sleep(30)
        btn.stop()

    asyncio.run(test())
