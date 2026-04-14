import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

print("Testing Forward...")
ser.write(b'F')
time.sleep(2)

print("Testing Stop...")
ser.write(b'S')
time.sleep(1)

print("Testing Left...")
ser.write(b'L')
time.sleep(2)

print("Testing Stop...")
ser.write(b'S')
time.sleep(1)

print("Testing Right...")
ser.write(b'R')
time.sleep(2)

print("Testing Stop...")
ser.write(b'S')

print("Done!")
ser.close()
