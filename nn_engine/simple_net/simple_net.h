#pragma once

#include <vector>
#include <memory>

#include "./simple_layer.h"
#include "./simple_activation_layer.h"

struct SimpleNet {

    std::vector<std::unique_ptr<AbstractSimpleLayer>> layers;

    // Default constructor
    SimpleNet() = default;

    // 1. Custom Deep Copy Constructor (using addLayer)
    SimpleNet(const SimpleNet& other) {
        layers.reserve(other.layers.size());
        for (const auto& layer : other.layers) {
            addLayer(layer->clone()); 
        }
    }

    // 2. Custom Deep Copy Assignment Operator (Copy-and-Swap idiom)
    SimpleNet& operator=(const SimpleNet& other) {
        if (this != &other) {
            SimpleNet temp(other);
            std::swap(layers, temp.layers);
        }
        return *this;
    }

    // Move constructor and move assignment
    SimpleNet(SimpleNet&&) noexcept = default;
    SimpleNet& operator=(SimpleNet&&) noexcept = default;

    void addLayer(std::unique_ptr<AbstractSimpleLayer> newLayer) {
        layers.push_back(std::move(newLayer));
    }

    std::vector<float> predict(const std::vector<float>& input) {
        std::vector<float> output = input;

        for(const auto& l : layers) {
            output = l->forward(output);
        }

        return output;
    }

    void train() {}
};