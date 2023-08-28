#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "types.h"

typedef struct neuron {
    f32* weights;
    f32 bias;
} neuron;

typedef struct layerDense {
    u32 n_inputs, n_neurons;
    neuron* neurons;
} layerDense;


f32 random_f() {
    f32 r = (f32)rand() - ((f32)RAND_MAX / 2);
    r /= ((f32)RAND_MAX / 2);

    return r;
}

void linspace(f32 start, f32 stop, u32 samples, f32 output[]) {
    f32 space = (stop - start) / (samples - 1);
    for (usize i = 0; i < samples; i++) {
        output[i] = start + space * i;
    }
}

/*
 * Adapted from: https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py 
 */
void spiral_data(u32 samples, u32 classes, f32 outputData[][2], u8 outputClasses[]) {
    for (usize class_index = 0; class_index < classes; class_index++) {
        f32 r[samples]; // radius
        f32 t[samples]; // theta
        linspace(0, 1, samples, r);
        linspace(class_index*4, (class_index+1)*4, samples, t);

        for (usize sample_index = 0; sample_index < samples; sample_index++) {
            r[sample_index] += random_f() * 0.2;

            outputData[sample_index + (class_index * samples)][0] = 
                r[sample_index]*sinf(t[sample_index]*2.5);

            outputData[sample_index + (class_index * samples)][1] = 
                r[sample_index]*cosf(t[sample_index]*2.5);

            outputClasses[sample_index + (class_index * samples)] = class_index;
        }
    }
}


/*
 * Adapted from: https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/vertical.py 
 */
void vertical_data(u32 samples, u32 classes, f32 outputData[][2], u8 outputClasses[]) {
    for (usize class_index = 0; class_index < classes; class_index++) {

        for (usize sample_index = 0; sample_index < samples; sample_index++) {
            outputData[sample_index + (class_index * samples)][0] = 
                random_f()*0.1 + ((f32)class_index)/3;

            outputData[sample_index + (class_index * samples)][1] = 
                random_f()*0.1 + 0.5;

            outputClasses[sample_index + (class_index * samples)] = class_index;
        }
    }
}

/* 
 * @brief takes two equally sized arrays and returns the dot product
 */
f32 vector_dot(f32 vector1[], f32 vector2[], usize size) {
    f32 accumulator = 0;
    for (usize i = 0; i < size; i++) {
        accumulator += vector1[i] * vector2[i];
    }

    return accumulator;
}

void matrix_vector_dot(usize rows, usize cols, f32 matrix[][cols], f32 vector[], f32 result[]) {
    for (usize i = 0; i < rows; i++) {
        result[i] = vector_dot(matrix[i], vector, cols);
    }
}

/* 
 * @brief takes two equally sized arrays and returns the result 
 * of their addition in vector1
 */
void vector_add(f32 vector1[], f32 vector2[], usize size) {
   for (usize i = 0; i < size; i++) {
       vector1[i] = vector1[i] + vector2[i];
   } 
}

void relu(f32 input[], usize size) {
    for (usize i = 0; i < size; i++) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}

f32 max_element(f32 input[], usize size) {
    f32 max = input[0];
    for (usize i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    return max;
}

f32 clip(f32 value, f32 min, f32 max) {
    if (value < min) {
        return min;
    }

    if (value > max) {
        return max;
    }

    return value;
}

f32 mean(f32 values[], usize size) {
    f32 accumulator = 0;
    for (usize i = 0; i < size; i++) {
        accumulator += values[i];
    }

    return accumulator / size;
}

void softmax(f32 input[], usize size) {
    // first we calculate the exponential value
    // and the sum of all exponentiated values
    f32 sum = 0;
    f32 max = max_element(input, size);
    for (usize i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }

    // Now we normalize
    for (usize i = 0; i < size; i++) {
        input[i] = input[i] / sum;
    }

}

void layerDense_init(layerDense* layer, u32 n_inputs, u32 n_neurons) {

    layer->n_inputs = n_inputs;
    layer->n_neurons = n_neurons;
    layer->neurons = malloc(sizeof(neuron) * n_neurons);
    for (usize i = 0; i < n_neurons; i++) {
        layer->neurons[i].bias = 0; 
        layer->neurons[i].weights = malloc(sizeof(f32) * n_inputs);

        for (usize j = 0; j < n_inputs; j++) {
            f32 weight = 0.01 * random_f();
            layer->neurons[i].weights[j] = weight;
        }
    }
}

void layerDense_forward(layerDense* layer, f32 inputs[], f32 output[]) {
    for (usize i = 0; i < layer->n_neurons; i++) {
        output[i] = 
            vector_dot(inputs, layer->neurons[i].weights, layer->n_inputs) +
            layer->neurons[i].bias;
    }
}

void layerDense_destroy(layerDense* layer) {
    for (usize i = 0; i < layer->n_neurons; i++) {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
}

void layerDense_print(layerDense* l) {
    for (usize neuron_index = 0; neuron_index < l->n_neurons; neuron_index++) {
        for (usize weight_index = 0; weight_index < l->n_inputs; weight_index++) {
            printf("%f - ", l->neurons[neuron_index].weights[weight_index]);
        }
        printf("\n");
    }
}

f32 loss_CategoricalCrossentropy(f32 prediction[], usize correctIndex) {
    f32 pred = clip(prediction[correctIndex], 1e-7, 1 - 1e-7);
    f32 loss = -logf(pred);
    return loss;
}


int main(int argc, char* argv[])
{
    srand(time(0));

    f32 outputData[300][2];
    u8 outputClasses[300];
    spiral_data(100, 3, outputData, outputClasses);

    layerDense dense1;
    layerDense_init(&dense1, 2, 3);

    layerDense dense2;
    layerDense_init(&dense2, 3, 3);

    f32 losses[300];
    for (usize i = 0; i < 300; i++) {
        f32 output[3];
        layerDense_forward(&dense1, outputData[i], output);
        relu(output, 3);
        layerDense_forward(&dense2, output, output);
        softmax(output, 3);

        losses[i] = loss_CategoricalCrossentropy(output, outputClasses[i]);
    }

    
    printf("%f", mean(losses, 300));
    return 0;
}

