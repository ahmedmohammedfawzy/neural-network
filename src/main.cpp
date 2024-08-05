#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

struct tData
{
    f32 posX;
    f32 posY;
    u8 label;
};

f32 random_f()
{
    f32 r = (f32)rand() - ((f32)RAND_MAX / 2);
    r /= ((f32)RAND_MAX / 2);

    return r;
}

/*
 * @brief takes two equally sized arrays and returns the dot product */
f32 vector_dot(vector<f32> const &vector1, vector<f32> const &vector2)
{
    f32 accumulator = 0;
    for (usize i = 0; i < vector1.size(); i++)
    {
        accumulator += vector1[i] * vector2[i];
    }

    return accumulator;
}
class neuron
{
  public:
    vector<f32> weights;
    f32 bias;

    neuron(u32 n_weights)
    {
        for (usize i = 0; i < n_weights; i++)
        {
            weights.push_back(0.05 * random_f());
        }
        bias = 0.05 * random_f();
    }
};

class layerDense
{
  public:
    u32 n_inputs;
    u32 n_neurons;
    vector<neuron> neurons;

    layerDense() {}

    layerDense(u32 n_inputs, u32 n_neurons)
    {
        this->n_inputs = n_inputs;
        this->n_neurons = n_neurons;
        for (usize i = 0; i < n_neurons; i++)
        {
            neurons.push_back(neuron(n_inputs));
        }
    }

    vector<f32> forward(vector<f32> const &inputs)
    {
        vector<f32> output;
        output.reserve(n_neurons);
        for (auto neuron : neurons)
        {
            f32 result = vector_dot(inputs, neuron.weights) + neuron.bias;
            output.push_back(result);
        }

        return output;
    }
};

vector<f32> linspace(f32 start, f32 stop, u32 samples)
{
    vector<f32> output(samples);
    f32 space = (stop - start) / (samples - 1);
    for (usize i = 0; i < samples; i++)
    {
        output[i] = start + space * i;
    }

    return output;
}

/*
 * Adapted from:
 * https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
 */
vector<tData> spiral_data(u32 samples, u32 classes)
{
    vector<tData> output(samples * classes);

    for (usize class_index = 0; class_index < classes; class_index++)
    {
        auto r = linspace(0, 1, samples);
        auto t = linspace(class_index * 4, (class_index + 1) * 4, samples);

        for (usize sample_index = 0; sample_index < samples; sample_index++)
        {
            r[sample_index] += random_f() * 0.2;

            usize index = sample_index + (class_index * samples);

            output[index].posX = r[sample_index] * sin(t[sample_index] * 2.5);
            output[index].posY = r[sample_index] * cos(t[sample_index] * 2.5);
            output[index].label = class_index;
        }
    }

    return output;
}

/*
 * Adapted from:
 * https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/vertical.py
 */
vector<tData> vertical_data(u32 samples, u32 classes)
{
    vector<tData> output(samples * classes);
    for (usize class_index = 0; class_index < classes; class_index++)
    {
        for (usize sample_index = 0; sample_index < samples; sample_index++)
        {
            usize index = sample_index + (class_index * samples);
            output[index].posX = random_f() * 0.1 + ((f32)class_index) / 3;
            output[index].posY = random_f() * 0.1 + 0.5;
            output[index].label = class_index;
        }
    }
    return output;
}

/*
 * @brief takes two equally sized arrays and returns the result
 * of their addition in vector1
 */
void vector_add(vector<f32> &vector1, vector<f32> &vector2)
{
    for (usize i = 0; i < vector1.size(); i++)
    {
        vector1[i] = vector1[i] + vector2[i];
    }
}

void relu(vector<f32> &input)
{
    for (usize i = 0; i < input.size(); i++)
    {
        if (input[i] < 0)
        {
            input[i] = 0;
        }
    }
}

f32 max_element(vector<f32> const &input)
{
    f32 max = input[0];
    for (usize i = 1; i < input.size(); i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    return max;
}

template <typename T> f32 mean(vector<T> const &values)
{
    f32 accumulator = 0;
    for (usize i = 0; i < values.size(); i++)
    {
        accumulator += values[i];
    }

    return accumulator / values.size();
}

void softmax(vector<f32> &input)
{
    // first we calculate the exponential value
    // and the sum of all exponentiated values
    f32 sum = 0;
    f32 max = max_element(input);
    for (usize i = 0; i < input.size(); i++)
    {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }

    // Now we normalize
    for (usize i = 0; i < input.size(); i++)
    {
        input[i] = input[i] / sum;
    }
}

f32 loss_CategoricalCrossentropy(vector<f32> const &prediction,
                                 usize correctIndex)
{
    f32 pred = clamp<f32>(prediction[correctIndex], 1e-7, 1 - 1e-7);
    f32 loss = -log(pred);
    return loss;
}

usize max_index(vector<f32> const &vector)
{
    f32 result = vector.at(0);
    usize index = 0;

    for (usize i = 0; i < vector.size(); i++)
    {
        if (vector[i] > result)
        {
            result = vector[i];
            index = i;
        }
    }

    return index;
}

layerDense best_dense1;
layerDense best_dense2;
f32 lowest_loss = 10000;

void run_batch(vector<tData> data)
{
    layerDense dense1(2, 3);
    layerDense dense2(3, 3);

    vector<f32> losses;
    vector<u8> correctness;
    losses.reserve(300);
    correctness.reserve(300);
    for (tData d : data)
    {
        vector<f32> output = dense1.forward({d.posX, d.posY});
        relu(output);
        output = dense2.forward(output);
        softmax(output);

        losses.push_back(loss_CategoricalCrossentropy(output, d.label));
        if (max_index(output) == d.label)
        {
            correctness.push_back(1);
        }
        else
        {
            correctness.push_back(0);
        }
    }

    f32 loss_mean = mean(losses);
    if (loss_mean < lowest_loss)
    {
        lowest_loss = loss_mean;
        best_dense1 = dense1;
        best_dense2 = dense2;
        cout << "loss: " << loss_mean << " acc: " << mean(correctness) << endl;
    }
}

int main(int argc, char *argv[])
{
    srand(time(0));

    auto data = vertical_data(100, 3);

    for (usize i = 0; i < 100000; i++)
    {
        run_batch(data);
    }

    return 0;
}
