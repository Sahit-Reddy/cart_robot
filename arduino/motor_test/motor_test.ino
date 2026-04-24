// Motor A (Left)
int enA = 5;
int in1 = 8;
int in2 = 9;

// Motor B (Right)
int enB = 6;
int in3 = 10;
int in4 = 11;

// Ultrasonic sensor
int trigPin = 22;
int echoPin = 23;

// Safety distance in cm
int safeDistance = 30;

void setup() {
  Serial.begin(9600);
  
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  stopMotors();
}

long getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  long duration = pulseIn(echoPin, HIGH, 30000);  // 30ms timeout
  long distance = duration * 0.034 / 2;  // Convert to cm
  
  if (distance == 0) {
    distance = 999;  // No reading = far away
  }
  
  return distance;
}

// Motor speed constants
const int FORWARD_SPEED = 120;
const int TURN_SPEED = 100;
const int TURN_PULSE_MS = 100;

void forward() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  analogWrite(enA, FORWARD_SPEED);
  analogWrite(enB, FORWARD_SPEED);
}

void backward() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  analogWrite(enA, FORWARD_SPEED);
  analogWrite(enB, FORWARD_SPEED);
}

void turnLeft() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  analogWrite(enA, TURN_SPEED);
  analogWrite(enB, TURN_SPEED);
  delay(TURN_PULSE_MS);
  stopMotors();
}

void turnRight() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  analogWrite(enA, TURN_SPEED);
  analogWrite(enB, TURN_SPEED);
  delay(TURN_PULSE_MS);
  stopMotors();
}

void stopMotors() {
  analogWrite(enA, 0);
  analogWrite(enB, 0);
}

void loop() {
  long distance = getDistance();
  
  // Safety override - stop if obstacle detected
  if (distance < safeDistance) {
    stopMotors();
    Serial.println("OBSTACLE");
    delay(100);
    return;  // Skip processing Pi commands
  }
  
  // Process commands from Pi
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == 'F') {
      forward();
      Serial.println("Forward");
    }
    else if (cmd == 'B') {
      backward();
      Serial.println("Backward");
    }
    else if (cmd == 'L') {
      turnLeft();
      Serial.println("Left");
    }
    else if (cmd == 'R') {
      turnRight();
      Serial.println("Right");
    }
    else if (cmd == 'S') {
      stopMotors();
      Serial.println("Stopped");
    }
  }
}