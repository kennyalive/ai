#include "random.h"
#include <algorithm>
#include <print>
#include <vector>
#include <random>

RNG rng;

float sigmoid_activation(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

float sigmoid_derivative(float sigmoid_value)
{
    return sigmoid_value * (1.f - sigmoid_value);
}

float get_init_value()
{
    return rng.get_float_signed() * 0.1f;
}

constexpr float R = 0.75f;

bool is_inside(float x, float y)
{
    return x * x + y * y <= R * R;
}

// Point inside/outside circle classification network.
// Two input neurons, four hidden neurons, one output neuron.
int main()
{
    float w0[2][4];
    float b0[4];
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 2; i++) {
            w0[i][k] = get_init_value();
        }
        b0[k] = get_init_value();
    }

    float w1[4];
    float b1;
    for (int i = 0; i < 4; i++) {
        w1[i] = get_init_value();
    }
    b1 = get_init_value();

    float v0[4];
    float v1;

    auto evaluate = [&](float x, float y) {
        for (int i = 0; i < 4; i++) {
            float k = w0[0][i] * x + w0[1][i] * y + b0[i];
            v0[i] = sigmoid_activation(k);
        }
        float m = b1;
        for (int i = 0; i < 4; i++) {
            m += w1[i] * v0[i];
        }
        v1 = sigmoid_activation(m);
        return v1;
    };

    float w0_grad[2][4];
    float b0_grad[4];
    float w1_grad[4];
    float b1_grad;
    auto backprop = [&](float x, float y, float loss_derivative) {
        float d1 = loss_derivative;
        for (int i = 0; i < 4; i++) {
            w1_grad[i] = d1 * v0[i];
        }
        b1_grad = d1;

        float d0[4];
        for (int i = 0; i < 4; i++) {
            d0[i] = d1 * w1[i] * sigmoid_derivative(v0[i]);
        }
        for (int i = 0; i < 4; i++) {
            w0_grad[0][i] = d0[i] * x;
            w0_grad[1][i] = d0[i] * y;
            b0_grad[i] = d0[i];
        }
    };

    const float learning_rate = 0.002f;
    auto update_weights = [&]() {
        for (int i = 0; i < 4; i++) {
            w0[0][i] -= learning_rate * w0_grad[0][i];
            w0[1][i] -= learning_rate * w0_grad[1][i];
            b0[i] -= learning_rate * b0_grad[i];
            w1[i] -= learning_rate * w1_grad[i];
        }
        b1 -= learning_rate * b1_grad;
    };

    const size_t training_size_random = 10'000;
    const size_t training_size_special = 14'000;
    std::vector<std::pair<float, float>> training_data(training_size_random + training_size_special);
    for (size_t i = 0; i < training_size_random; i++) {
        training_data[i] = { rng.get_float_signed(), rng.get_float_signed() };
    }
    for (size_t i = training_size_random; i < training_data.size(); i++) {
        float r = R + rng.get_float_signed() * 0.01f;
        float angle = 2 * 3.14159f * rng.get_float();
        float x = std::cos(angle) * r;
        float y = std::sin(angle)* r;
        training_data[i] = {x, y};
    }

    std::mt19937 g(13);
    std::shuffle(training_data.begin(), training_data.end(), g);

    for (int i = 0; i < 10'000; i++) {
        for (auto [x, y] : training_data) {
            float p0 = is_inside(x, y) ? 1.f : 0.f;
            float p = evaluate(x, y);
            backprop(x, y, p - p0);
            update_weights();
        }
    }

    const int N = 10'000;
    int X = 0;
    for (int i = 0; i < N; i++) {
        const float x = rng.get_float_signed();
        const float y = rng.get_float_signed();
        bool inside = is_inside(x, y) ? 1.f : 0.f;
        float p = evaluate(x, y);
        bool correct = inside ? p > 0.8f : p < 0.2f;
        X += correct;
        std::print("x = {: .3f}, y = {: .3f}, {:7}, prediction = {:.3f}, {}\n", x, y, inside ? "inside" : "outside", p, correct ? "CORRECT" : "NO");
    }
    std::print("accuracy = {:.2f}%\n", float(X) / float(N) * 100.f);
}
