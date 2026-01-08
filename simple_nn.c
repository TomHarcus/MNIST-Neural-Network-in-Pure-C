#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01

float input_vector[INPUT_SIZE];
float target[10];

float W1[HIDDEN_SIZE][INPUT_SIZE];
float B1[HIDDEN_SIZE];
float W2[OUTPUT_SIZE][HIDDEN_SIZE];
float B2[OUTPUT_SIZE];

float a1[HIDDEN_SIZE];

float z2[OUTPUT_SIZE];
float a2[OUTPUT_SIZE];

int file_exists(const char* filename) {
  struct stat buffer;
  return stat(filename, &buffer) == 0;
}

void load_mnist_image_label(const char* image_file, const char* label_file, int index, float* input_vector, float* target) {
  FILE* img = fopen(image_file, "rb");
  FILE* lbl = fopen(label_file, "rb");

  if (!img || !lbl) {
    printf("Error opening MNIST files\n");
    exit(1);
  }

  fseek(img, 16 + index * 784, SEEK_SET);
  fseek(lbl, 8 + index, SEEK_SET);

  unsigned char pixels[784];
  unsigned char label;

  fread(pixels, 1, 784, img);
  fread(&label, 1, 1, lbl);

  fclose(img);
  fclose(lbl);

  for (int i = 0; i< 784; i++) {
    input_vector[i] = pixels[i] / 255.0f;
  }

  for (int i = 0; i < 10; i++) {
    target[i] = (i == label) ? 1.0f : 0.0f;
  }
}


float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float dsigmoid(float x) {
  float s = sigmoid(x);
  return s * (1.0f - s);
}

void softmax(float* input, float* output, int size) {
  float sum = 0.0f;

  for (int i = 0; i < size; i++) {
    sum += expf(input[i]);
  }

  for (int i = 0; i < size; i++) {
    output[i] = expf(input[i]) / sum;
  }
  
}

void init_values() {

  // init W1 
  float limit = sqrtf(1.0f / INPUT_SIZE);

  for (int i = 0; i < HIDDEN_SIZE; i++) {
    for (int j = 0; j < INPUT_SIZE; j++) {
      float rand_unit = (float)rand() / RAND_MAX;
      W1[i][j] = (2.0f * rand_unit - 1.0f) * limit;
    }
  }

  // init B1
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    B1[i] = 0.0f;
  }
  
  // init W2
  
  float limit2 = sqrtf(1.0f / HIDDEN_SIZE);

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      float rand_unit = (float)rand() / RAND_MAX;
      W2[i][j] = (2.0f * rand_unit - 1.0f) * limit2;
    }
  }

  // init B2
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    B2[i] = 0;
  }
}

void forward_pass() {
  // input -> h1
  
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = 0.0f;
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += W1[i][j] * input_vector[j];
    }
    sum += B1[i];
    a1[i] = sigmoid(sum);
  }

  // h1 -> output
  
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    float sum = 0.0f;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += W2[i][j] * a1[j];
    }
    sum += B2[i];
    z2[i] = sum;
  }
  
  softmax(z2, a2, OUTPUT_SIZE);
  

}

void backprop() {
  float delta2[OUTPUT_SIZE];
  float delta1[HIDDEN_SIZE];

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    delta2[i] = a2[i] - target[i];
  }

  for (int j = 0; j < HIDDEN_SIZE; j++) {
    delta1[j] = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      delta1[j] += delta2[i] * W2[i][j];
    }
    delta1[j] *= a1[j] * (1.0f - a1[j]);
  }

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      W2[i][j] -= LEARNING_RATE * delta2[i] * a1[j];
    }
    B2[i] -= LEARNING_RATE * delta2[i];
  }

  for (int j = 0; j < HIDDEN_SIZE; j++) {
    for (int k = 0; k < INPUT_SIZE; k++) {
      W1[j][k] -= LEARNING_RATE * delta1[j] * input_vector[k];
    }
    B1[j] -= LEARNING_RATE * delta1[j];
  }
}

void print_image(float* input_vector) {
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      float pixel = input_vector[i * 28 + j];

      // Use ASCII grayscale levels
      if (pixel > 0.8f) {
        printf("@");
      } else if (pixel > 0.6f) {
        printf("#");
      } else if (pixel > 0.4f) {
        printf("*");
      } else if (pixel > 0.2f) {
        printf(".");
      } else {
        printf(" ");
      }
    }
    printf("\n");
  }
}

void save_parameters(const char* filename) {
  FILE* f = fopen(filename, "wb");
  if (!f) {
    printf("Failed to open file\n");
      exit(1);
  }

  fwrite(W1, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
  fwrite(B1, sizeof(float), HIDDEN_SIZE, f);
  fwrite(W2, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
  fwrite(B2, sizeof(float), OUTPUT_SIZE, f);

  fclose(f);
  printf("Parameters saved to '%s'\n", filename);
}

void load_parameters(const char* filename) {
  FILE* f = fopen(filename, "rb");
  if (!f) {
    printf("Failed to open file\n");
    exit(1);
  }

  fread(W1, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
  fread(B1, sizeof(float), HIDDEN_SIZE, f);
  fread(W2, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
  fread(B2, sizeof(float), OUTPUT_SIZE, f);

  fclose(f);
  printf("Parameters loaded from '%s'\n", filename);
}




int main() {
  const char* model_file = "model_epoch5.dat";

  srand(time(NULL));

  if (file_exists(model_file)) {
    printf("Loading trained model...\n");
    load_parameters(model_file);

    int sample = rand() % (60000 - 10);

    for (int i = sample; i < sample+3; i++) {
      load_mnist_image_label("train-images-idx3-ubyte", "train-labels-idx1-ubyte", i, input_vector, target);
      forward_pass();

      printf("\nSample #%d\n", i);
      print_image(input_vector);

      // Print predicted class
      int predicted = 0;
      float max_prob = a2[0];
      for (int j = 1; j < OUTPUT_SIZE; j++) {
        if (a2[j] > max_prob) {
          max_prob = a2[j];
          predicted = j;
        }
      }

      // Find actual label
      int actual = 0;
      for (int j = 0; j < OUTPUT_SIZE; j++) {
        if (target[j] == 1.0f) {
          actual = j;
          break;
        }
      }

    printf("Predicted: %d (%.2f%% confidence)\n", predicted, max_prob * 100);
    printf("Actual:    %d\n", actual);
    printf("\n");
  }

  } else {
    printf("Training model from scratch...\n");

    int epochs = 5;
    int samples = 60000;

    init_values();

    for (int epoch = 0; epoch < epochs; epoch++) {
      printf("Epoch %d\n", epoch + 1);

      for (int i = 0; i < samples; i++) {
        load_mnist_image_label("train-images-idx3-ubyte", "train-labels-idx1-ubyte", i, input_vector, target);
        forward_pass();
        backprop();

        if (i % 1000 == 0) {
          printf("  Trained on %d samples\n", i);
        }
      }
    }

    save_parameters(model_file);
    }

  return 0;
}

