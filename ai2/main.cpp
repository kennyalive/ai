#include "random.h"
#include <algorithm>
#include <print>
#include <vector>

RNG rng;

float sigmoid_activation(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

float get_init_value()
{
    return rng.get_float_signed() * 0.1f;
}

// XOR evaluation network.
// Two input neurons, two hidden neurons, one output neuron.
// Hiden and output neurons use sigmoid activation.
int main()
{
    float w0 = get_init_value();
    float w1 = get_init_value();
    float b0 = get_init_value();

    float w2 = get_init_value();
    float w3 = get_init_value();
    float b1 = get_init_value();

    float w4 = get_init_value();
    float w5 = get_init_value();
    float b2 = get_init_value();

    const float learning_rate = 1.f;

    for (int i = 0; i < 10000; i++) {
        const int a = int(rng.get_float() + 0.5f);
        const int b = int(rng.get_float() + 0.5f);

        const float y0 = float(a ^ b);

        // evaluate network
        const float x0 = w0*a + w1*b + b0;
        const float c = sigmoid_activation(x0); // activation function

        const float x1 = w2*a + w3*b + b1;
        const float d = sigmoid_activation(x1);

        const float x2 = w4*c + w5*d + b2;
        const float y = sigmoid_activation(x2); // predicted value

        // compute loss derivatives
        float dLoss_dy = 2.f * (y - y0);
        float dy_dx2 = y * (1.f - y);

        float dLoss_dw4 = dLoss_dy * dy_dx2 * c;
        float dLoss_dw5 = dLoss_dy * dy_dx2 * d;
        float dLoss_db2 = dLoss_dy * dy_dx2;

        float dy_dc = dy_dx2 * w4;
        float dc_dx0 = c * (1.f - c);
        float dLoss_dw0 = dLoss_dy * dy_dc * dc_dx0 * a;
        float dLoss_dw1 = dLoss_dy * dy_dc * dc_dx0 * b;
        float dLoss_db0 = dLoss_dy * dy_dc * dc_dx0;

        float dy_dd = dy_dx2 * w5;
        float dd_dx1 = d * (1.f - d);
        float dLoss_dw2 = dLoss_dy * dy_dd * dd_dx1 * a;
        float dLoss_dw3 = dLoss_dy * dy_dd * dd_dx1 * b;
        float dLoss_db1 = dLoss_dy * dy_dd * dd_dx1;

        // update weights
        w0 -= learning_rate * dLoss_dw0;
        w1 -= learning_rate * dLoss_dw1;
        w2 -= learning_rate * dLoss_dw2;
        w3 -= learning_rate * dLoss_dw3;
        w4 -= learning_rate * dLoss_dw4;
        w5 -= learning_rate * dLoss_dw5;

        b0 -= learning_rate * dLoss_db0;
        b1 -= learning_rate * dLoss_db1;
        b2 -= learning_rate * dLoss_db2;
    }

    auto evaluate = [&](int af, int bf) {
        int a = int(af);
        int b = int(bf);
        float c = sigmoid_activation(w0*a + w1*b + b0);
        float d = sigmoid_activation(w2*a + w3*b + b1);
        float y = sigmoid_activation(w4*c + w5*d + b2);
        return y;
    };

    std::print("0 ^ 0 = 0, prediction = {}\n", evaluate(0, 0));
    std::print("0 ^ 1 = 1, prediction = {}\n", evaluate(0, 1));
    std::print("1 ^ 0 = 1, prediction = {}\n", evaluate(1, 0));
    std::print("1 ^ 1 = 0, prediction = {}\n", evaluate(1, 1));
}
