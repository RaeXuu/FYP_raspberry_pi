import RPi.GPIO as GPIO
import time

PIN_BUZZER = 25

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_BUZZER, GPIO.OUT, initial=GPIO.HIGH)  # 默认HIGH=静音

print("短响一声...")
GPIO.output(PIN_BUZZER, GPIO.LOW)
time.sleep(0.1)
GPIO.output(PIN_BUZZER, GPIO.HIGH)
time.sleep(0.5)

print("连响三声...")
for _ in range(3):
    GPIO.output(PIN_BUZZER, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(PIN_BUZZER, GPIO.HIGH)
    time.sleep(0.2)

time.sleep(0.5)
print("长响一声...")
GPIO.output(PIN_BUZZER, GPIO.LOW)
time.sleep(0.8)
GPIO.output(PIN_BUZZER, GPIO.HIGH)

GPIO.cleanup([PIN_BUZZER])
print("完成")
