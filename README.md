# Autonomous Shopping Cart Retrieval Robot

A prototype robot that autonomously retrieves shopping carts from a parking lot and returns them to a designated location using computer vision and AprilTag navigation.

## Overview

This project uses a Kinect sensor to detect AprilTags for localization and navigation. The robot follows tags to navigate from a pickup zone to a dropoff zone while avoiding obstacles.

## Hardware

- Raspberry Pi 3B+
- Microsoft Kinect v1 (for Windows)
- Arduino Mega 2560
- L298N Motor Driver
- 12V DC Motors with Encoders
- 12V Power Source
- Mini Shopping Cart (test platform)

## Software

- Python 3 with OpenCV
- AprilTag library for tag detection
- Freenect for Kinect interface
- PySerial for Pi-Arduino communication

## How It Works

1. Kinect RGB camera detects AprilTags in the scene
2. Pi calculates tag position (left, center, right) and distance
3. Pi sends motor commands to Arduino (Forward, Left, Right, Stop)
4. Arduino controls motors via L298N driver
5. Robot moves toward tag until it reaches target distance
6. Robot stops when it arrives at destination

## Files

- `follow_tag.py` - Main navigation script with visual feedback
- `test_apriltag.py` - AprilTag detection test
- `test_motors.py` - Motor control test
- `test_serial.py` - Pi-Arduino communication test
- `arduino/motor_test/motor_test.ino` - Arduino motor control code

## Usage

1. Print AprilTags (tag36h11 family, IDs 0 and 1)
2. Place Tag 0 at start location, Tag 1 at destination
3. Power on the system
4. Run: `python3 follow_tag.py`
