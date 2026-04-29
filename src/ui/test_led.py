import RPi.GPIO as GPIO
import time

PIN_R = 17
PIN_G = 22
PIN_B = 10

GPIO.setmode(GPIO.BCM)
for pin in (PIN_R, PIN_G, PIN_B):
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

pwm_r = GPIO.PWM(PIN_R, 100)
pwm_g = GPIO.PWM(PIN_G, 100)
pwm_b = GPIO.PWM(PIN_B, 100)
pwm_r.start(0)
pwm_g.start(0)
pwm_b.start(0)

def set_rgb(r, g, b):
    pwm_r.ChangeDutyCycle(r)
    pwm_g.ChangeDutyCycle(g)
    pwm_b.ChangeDutyCycle(b)

BRIGHTNESS = 0.3  # 调整这里，0.0~1.0

tests = [
    ("红",   100,   0,   0),
    ("绿",     0, 100,   0),
    ("蓝",     0,   0, 100),
    ("黄",   100, 100,   0),
    ("青",     0, 100, 100),
    ("白",   100, 100, 100),
    ("橙",   100,  30,   0),
    ("灭",     0,   0,   0),
]
tests = [(n, r*BRIGHTNESS, g*BRIGHTNESS, b*BRIGHTNESS) for n, r, g, b in tests]

try:
    for name, r, g, b in tests:
        print(f"{name}  R={r} G={g} B={b}")
        set_rgb(r, g, b)
        time.sleep(1)
finally:
    pwm_r.stop()
    pwm_g.stop()
    pwm_b.stop()
    GPIO.cleanup([PIN_R, PIN_G, PIN_B])
    print("完成")
