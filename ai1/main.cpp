#include "random.h"
#include <algorithm>
#include <print>
#include <vector>

const float scale = 5.f;

float sigmoid_activation(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

float loss_derivative_weight(float coordinate, int label_class, float predicted_class)
{
    // dL/dw for cross entropy loss
    return (predicted_class - (float)label_class) * coordinate;
}

float loss_derivative_bias(int label_class, float predicted_class)
{
    // dL/db for cross entropy loss
    return (predicted_class - (float)label_class);
}

int get_point_class(float x, float y)
{
    return (y < 0.5f * x + 1) ? 0 : 1;
}

struct Point
{
    float x;
    float y;
};

float coord(RNG& rng) { return scale * (2.f * rng.get_float() - 1.f); }

std::vector<Point> get_training_points(RNG& rng)
{
    std::vector<Point> p;
    for (int i = 0; i < 100; i++) {
        p.emplace_back(Point{ coord(rng), coord(rng) });
    }
    return p;
}

std::vector<Point> get_test_points(RNG& rng)
{
    std::vector<Point> p;
    for (int i = 0; i < 10; i++) {
        p.emplace_back(Point{ coord(rng), coord(rng) });
    }
    return p;
}

int main()
{
    RNG rng;

    const auto test_points = get_test_points(rng);
    const auto training_points = get_training_points(rng);

    float w1 = 2 * rng.get_float() - 1;
    float w2 = 2 * rng.get_float() - 1;
    float bias = 2 * rng.get_float() - 1;
    float learning_rate = 5.f;

    auto print_test_results = [&]() {
        for (size_t i = 0; i < test_points.size(); i++) {
            float x = test_points[i].x;
            float y = test_points[i].y;
            int label_class = get_point_class(x, y);

            x /= scale;
            y /= scale;
            float predicted_class = sigmoid_activation(x * w1 + y * w2 + bias);
            std::print("test point #{}: label_class {} predicted_class {}\n", i, label_class, predicted_class);
        }
    };

    std::print("Initial test results\n");
    print_test_results();

    for (int epoch = 0; epoch < 10; epoch++) {
        for (const Point& p : training_points) {
            float x = p.x;
            float y = p.y;
            int label_class = get_point_class(x, y);

            x /= scale;
            y /= scale;
            float predicted_class = sigmoid_activation(x * w1 + y * w2 + bias);

            float loss_derivative_w1 = loss_derivative_weight(x, label_class, predicted_class);
            float loss_derivative_w2 = loss_derivative_weight(y, label_class, predicted_class);
            float loss_derivative_b = loss_derivative_bias(label_class, predicted_class);

            w1 = w1 - learning_rate * loss_derivative_w1;
            w2 = w2 - learning_rate * loss_derivative_w2;
            bias = bias - learning_rate * loss_derivative_b;
        }
        std::print("\nTest results after epoch {}\n", epoch);
        print_test_results();
    }
    std::print("Trained values: w1 = {} w2 = {} bias = {}\n", w1, w2, bias);
}
