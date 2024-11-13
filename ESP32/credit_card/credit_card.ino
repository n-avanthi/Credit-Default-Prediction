#include <EloquentTinyML.h>
#include "credit_card_model.h"

// Model parameters
#define NUMBER_OF_INPUTS 23         // Number of features (after dropping ID)
#define NUMBER_OF_OUTPUTS 1         // Binary classification
#define TENSOR_ARENA_SIZE 8 * 1024  
#define NUM_SAMPLES 4               // Number of input samples to process

// StandardScaler parameters calculated from training data
struct ScalerParams {
  float mean;
  float scale;
};

ScalerParams scaler_params[NUMBER_OF_INPUTS] = {
  // LIMIT_BAL
  { 167484.3230, 129745.4990 },
  // SEX
  { 1.6037, 0.4912 },
  // EDUCATION
  { 1.8531, 0.7903 },
  // MARRIAGE
  { 1.5519, 0.5220 },
  // AGE
  { 35.4855, 9.2178 },
  // PAY_0 to PAY_6
  { -0.0167, 1.1238 },
  { -0.1338, 1.1972 },
  { -0.1662, 1.1968 },
  { -0.2207, 1.1691 },
  { -0.2662, 1.1332 },
  { -0.2911, 1.1500 },
  // BILL_AMT1 to BILL_AMT6
  { 51223.3309, 73634.6333 },
  { 49179.0752, 71172.5825 },
  { 47013.1548, 69348.2316 },
  { 43262.9490, 64331.7839 },
  { 40311.4010, 60796.1425 },
  { 38871.7604, 59553.1150 },
  // PAY_AMT1 to PAY_AMT6
  { 5663.5805, 16563.0043 },
  { 5921.1635, 23040.4864 },
  { 5225.6815, 17606.6680 },
  { 4826.0769, 15665.8986 },
  { 4799.3876, 15278.0510 },
  { 5215.5026, 17777.1695 }
};

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

// Function to preprocess a single feature
float preprocessFeature(float value, int feature_index) {
  return (value - scaler_params[feature_index].mean) / scaler_params[feature_index].scale;
}

// Function to process raw input data
void preprocessInput(float* raw_input, float* scaled_input) {
  // Skip the ID column from raw_input
  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    scaled_input[i] = preprocessFeature(raw_input[i + 1], i);  // +1 to skip ID
  }
}

void setup() {
  Serial.begin(115200);

  while (!Serial) {
    ;  // Wait for serial port to connect
  }

  Serial.println("Initializing Credit Card Default Prediction Model...");

  if (ml.begin(credit_card_model)) {
    Serial.println("Model loaded successfully");
  } else {
    Serial.println("Failed to load model");
    while (1)
      ;  // Halt if model fails to load
  }
}

float scaled_input[NUM_SAMPLES][NUMBER_OF_INPUTS];
float output[NUM_SAMPLES][NUMBER_OF_OUTPUTS];

float raw_inputs[NUM_SAMPLES][NUMBER_OF_INPUTS] = {0};

void loop() {
  // Sample inputs from your Python code (including ID)
  float sample1[] = { 14, 70000, 1, 2, 2, 30, 1, 2, 2, 0, 0, 2, 65802, 67369, 65701, 66782, 36137, 36894, 3200, 0, 3000, 3000, 1500, 0, 1 };
  float sample2[] = { 13, 630000, 2, 2, 2, 41, -1, 0, -1, -1, -1, -1, 12137, 6500, 6500, 6500, 6500, 2870, 1000, 6500, 6500, 6500, 2870, 0, 0 };
  float sample3[] = { 16, 50000, 2, 3, 3, 23, 1, 2, 0, 0, 0, 0, 50614, 29173, 28116, 28771, 29531, 30211, 0, 1500, 1100, 1200, 1300, 1100, 0 };
  float sample4[] = { 17, 20000, 1, 1, 2, 24, 0, 0, 2, 2, 2, 2, 15376, 18010, 17428, 18338, 17905, 19104, 3200, 0, 1500, 0, 1650, 0, 1 };

  for (int i = 0; i < NUM_SAMPLES; i++) {
    // Copy the feature data (excluding ID) to the raw_inputs array
    for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
      raw_inputs[i][j] = i == 0 ? sample1[j] : (i == 1 ? sample2[j] : (i == 2 ? sample3[j] : sample4[j]));
    }

    // Preprocess the input
    preprocessInput(raw_inputs[i], scaled_input[i]);

    // Print scaled inputs for debugging
    // Serial.printf("Scaled inputs for sample %d:\n", i + 1);
    // for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
    //   Serial.print(scaled_input[i][j], 6);
    //   Serial.print(", ");
    //   if ((j + 1) % 5 == 0) Serial.println();
    // }
    // Serial.println();
  }

  // Make predictions
  for (int i = 0; i < NUM_SAMPLES; i++) {
    if (ml.predict(scaled_input[i], output[i])) {
      float probability = output[i][0];
      Serial.printf("Default probability for sample %d: %.6f\n", i + 1, probability);

      // Convert to class prediction
      int predicted_class = probability > 0.5f ? 1 : 0;
      Serial.printf("Predicted class for sample %d: %d\n", i + 1, predicted_class);

      // Print the expected output
      int expected_output = i == 0 ? 1 : (i == 1 ? 0 : (i == 2 ? 0 : 1));
      Serial.printf("Expected output for sample %d: %d\n", i + 1, expected_output);
    } else {
      Serial.printf("Prediction failed for sample %d\n", i + 1);
    }
  }

  delay(5000);  // Wait 5 seconds before next prediction
}