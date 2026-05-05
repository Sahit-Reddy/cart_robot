# Jetson Setup Guide

Dependencies and setup instructions for running the cart robot on NVIDIA Jetson.

## Arduino CLI

```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc
source ~/.bashrc
arduino-cli core install arduino:avr
```

## Kinect (libfreenect)

```bash
sudo apt-get update
sudo apt-get install -y libfreenect-dev freenect python3-pip
pip3 install freenect
```

## Python Libraries

```bash
pip3 install numpy opencv-python pyserial apriltag
```

## YOLO

### Option 1: YOLOv8 (recommended - easier setup)

```bash
pip3 install ultralytics
```

### Option 2: YOLOv5

```bash
pip3 install torch torchvision
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip3 install -r requirements.txt
```

## System Configuration

### Blacklist conflicting Kinect module

```bash
echo "blacklist gspca_kinect" | sudo tee /etc/modprobe.d/blacklist-kinect.conf
```

### Add user to required groups

```bash
sudo usermod -a -G video $USER
sudo usermod -a -G dialout $USER
```

### Reboot

```bash
sudo reboot
```

## Verify Installation

### Test Kinect

```bash
freenect-glview
```

### Test Arduino connection

```bash
ls /dev/ttyACM*
```

### Test YOLO

```bash
python3 -c "from ultralytics import YOLO; print('YOLO ready')"
```

### Test AprilTag

```bash
python3 -c "import apriltag; print('AprilTag ready')"
```

## Running the Robot

### Upload Arduino code

```bash
cd ~/cart_robot
arduino-cli compile --fqbn arduino:avr:mega arduino/motor_test
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:mega arduino/motor_test
```

### Run tag follower

```bash
python3 ~/cart_robot/follow_tag_new.py
```

## Troubleshooting

### Kinect not detected

1. Unplug and replug Kinect (both USB and power)
2. Run `sudo modprobe -r gspca_kinect`
3. Check with `lsusb | grep Xbox`

### Arduino permission denied

```bash
sudo chmod 666 /dev/ttyACM0
```

### Serial port not found

Check available ports:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

Update the port in follow_tag_new.py if needed.
