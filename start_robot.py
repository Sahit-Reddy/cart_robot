import RPi.GPIO as GPIO
import os
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Waiting for button press...")

while True:
    if GPIO.input(17) == GPIO.LOW:
        print("Starting robot...")
        os.system("python3 /home/pi504/cart_robot/follow_tag_new.py")
        break
    time.sleep(0.1)