#include <Servo.h>

// Define the servo motor pin
int servoPin = 9;

// Create a servo object
Servo servo;

void setup() {
  // Initialize the servo motor
  servo.attach(servoPin);
  
  // Initialize the serial communication
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    // Read the input value from the serial port
    int input = Serial.parseInt();
    
    // Map the input value to the servo angle
    int angle = map(input, 0, 1023, 0, 180);
    
    // Set the servo angle
    servo.write(angle);
  }
}
