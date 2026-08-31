#include <Arduino_BMI270_BMM150.h>

const unsigned long SAMPLE_INTERVAL_US = 20000;  // 50 samples per second.
unsigned long previousSampleUs = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {
  }

  if (!IMU.begin()) {
    Serial.println("ERROR: IMU initialization failed");
    while (true) {
    }
  }

  Serial.println("ax,ay,az,gx,gy,gz");
}

void loop() {
  const unsigned long now = micros();
  if (now - previousSampleUs < SAMPLE_INTERVAL_US) {
    return;
  }
  previousSampleUs = now;

  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    return;
  }

  float ax, ay, az;
  float gx, gy, gz;
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  Serial.print(ax, 6);
  Serial.print(',');
  Serial.print(ay, 6);
  Serial.print(',');
  Serial.print(az, 6);
  Serial.print(',');
  Serial.print(gx, 6);
  Serial.print(',');
  Serial.print(gy, 6);
  Serial.print(',');
  Serial.println(gz, 6);
}
