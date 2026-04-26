// Motor PWM Control
// Accepts commands in format: M,<left_pwm>,<right_pwm>\n
// PWM values can be negative for reverse

// Motor A (Left)
int enA = 5;
int in1 = 8;
int in2 = 9;

// Motor B (Right)
int enB = 6;
int in3 = 10;
int in4 = 11;

// Serial buffer
String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  
  stopMotors();
  Serial.println("Motor PWM Control Ready");
  Serial.println("Format: M,<left_pwm>,<right_pwm>");
}

void setMotorA(int pwm) {
  if (pwm > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(enA, pwm);
  } else if (pwm < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(enA, -pwm);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(enA, 0);
  }
}

void setMotorB(int pwm) {
  if (pwm > 0) {
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    analogWrite(enB, pwm);
  } else if (pwm < 0) {
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    analogWrite(enB, -pwm);
  } else {
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
    analogWrite(enB, 0);
  }
}

void stopMotors() {
  setMotorA(0);
  setMotorB(0);
}

void parseAndExecute(String command) {
  command.trim();
  
  // Check for M command: M,<left>,<right>
  if (command.startsWith("M,")) {
    int firstComma = command.indexOf(',');
    int secondComma = command.indexOf(',', firstComma + 1);
    
    if (firstComma > 0 && secondComma > firstComma) {
      String leftStr = command.substring(firstComma + 1, secondComma);
      String rightStr = command.substring(secondComma + 1);
      
      int leftPwm = leftStr.toInt();
      int rightPwm = rightStr.toInt();
      
      // Clamp to valid PWM range
      leftPwm = constrain(leftPwm, -255, 255);
      rightPwm = constrain(rightPwm, -255, 255);
      
      setMotorA(leftPwm);
      setMotorB(rightPwm);
    }
  }
  // Also support simple commands for backwards compatibility
  else if (command == "S") {
    stopMotors();
  }
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n') {
      parseAndExecute(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}