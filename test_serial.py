import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # wait for Arduino to reset

print("Sending 'F' (forward)...")
ser.write(b'F')
time.sleep(1)
response = ser.readline().decode().strip()
print(f"Arduino says: {response}")

time.sleep(2)

print("Sending 'S' (stop)...")
ser.write(b'S')
time.sleep(1)
response = ser.readline().decode().strip()
print(f"Arduino says: {response}")

ser.close()
