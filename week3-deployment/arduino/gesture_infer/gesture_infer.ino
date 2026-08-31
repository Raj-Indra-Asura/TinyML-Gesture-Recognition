#include <Arduino_BMI270_BMM150.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

constexpr int kWindowSamples = 128;
constexpr int kFeatureCount = 6;
constexpr unsigned long kSampleIntervalUs = 20000;
constexpr float kConfidenceThreshold = 0.70f;
constexpr int kTensorArenaSize = 120 * 1024;

alignas(16) byte tensorArena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int sampleIndex = 0;
unsigned long previousSampleUs = 0;

int8_t quantizeInput(float value) {
  const float scale = input->params.scale;
  const int32_t zeroPoint = input->params.zero_point;
  long quantized = lroundf(value / scale) + zeroPoint;
  quantized = constrain(quantized, -128, 127);
  return static_cast<int8_t>(quantized);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {
  }

  if (!IMU.begin()) {
    Serial.println("ERROR: IMU initialization failed");
    while (true) {
    }
  }

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: model schema version is unsupported");
    while (true) {
    }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, kTensorArenaSize);
  interpreter = &staticInterpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: tensor arena is too small");
    while (true) {
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    Serial.println("ERROR: expected a fully int8 model");
    while (true) {
    }
  }
  Serial.println("Ready. Perform one gesture for about 2.5 seconds.");
}

void loop() {
  const unsigned long now = micros();
  if (now - previousSampleUs < kSampleIntervalUs) {
    return;
  }
  previousSampleUs = now;
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    return;
  }

  float values[kFeatureCount];
  IMU.readAcceleration(values[0], values[1], values[2]);
  IMU.readGyroscope(values[3], values[4], values[5]);
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    const float normalized =
        (values[feature] - kNormalizationMean[feature]) /
        kNormalizationStd[feature];
    const int offset = sampleIndex * kFeatureCount + feature;
    input->data.int8[offset] = quantizeInput(normalized);
  }
  ++sampleIndex;
  if (sampleIndex < kWindowSamples) {
    return;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: inference failed");
    sampleIndex = 0;
    return;
  }

  int bestIndex = 0;
  float bestProbability = 0.0f;
  for (int index = 0; index < kGestureCount; ++index) {
    const float probability =
        (output->data.int8[index] - output->params.zero_point) *
        output->params.scale;
    if (probability > bestProbability) {
      bestProbability = probability;
      bestIndex = index;
    }
  }

  if (bestProbability >= kConfidenceThreshold) {
    Serial.print(kGestureLabels[bestIndex]);
  } else {
    Serial.print("uncertain");
  }
  Serial.print(", confidence=");
  Serial.println(bestProbability, 3);
  sampleIndex = 0;
}
