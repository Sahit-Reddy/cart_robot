// Motor A (Left)
int enA = 5;
int in1 = 8;
int in2 = 9;

// Motor B (Right)
int enB = 6;
int in3 = 10;
int in4 = 11;

void setup() {
  Serial.begin(9600);
  
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  
  stopMotors();
}

void forward() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  analogWrite(enA, 150);
  analogWrite(enB, 150);
}

void backward() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  analogWrite(enA, 150);
  analogWrite(enB, 150);
}

void turnLeft() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  analogWrite(enA, 150);
  analogWrite(enB, 150);
}

void turnRight() {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  analogWrite(enA, 150);
  analogWrite(enB, 150);
}

void stopMotors() {
  analogWrite(enA, 0);
  analogWrite(enB, 0);
}

void loop() {
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
