#include <assert.h>
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
    srand(time(0));

    layer->n_inputs = n_inputs;
    layer->n_neurons = n_neurons;
    layer->neurons = malloc(sizeof(neuron) * n_neurons);
    for (usize i = 0; i < n_neurons; i++) {
        layer->neurons[i].bias = 0; 
        layer->neurons[i].weights = malloc(sizeof(f32) * n_inputs);

        for (usize j = 0; j < n_inputs; j++) {
            f32 weight = (f32)rand() - ((f32)RAND_MAX / 2);
            weight /= ((f32)RAND_MAX / 2);
            weight *= 0.01;
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

    relu(output, layer->n_neurons);
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

int main(int argc, char* argv[])
{
    layerDense l;
    layerDense_init(&l, 2, 4);
    layerDense_print(&l);

    f32 output[l.n_neurons];
    f32 input[] = {2, 2};
    layerDense_forward(&l, input, output);

    for (usize i = 0; i < l.n_neurons; i++) {
        printf("%f\n", output[i]);
    }

    layerDense_destroy(&l);


    f32 softmax_inputs[] = {4.8, 1.21, 2.385};
    softmax(softmax_inputs, 3);

    f32 sum = 0;
    for (usize i = 0; i < 3; i++) {
        sum += softmax_inputs[i];
        printf("%f -", softmax_inputs[i]);
    }

    printf("%f", sum);
    return 0;
}

