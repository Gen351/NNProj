#pragma once

#include "./simple_layer.h"
#include<memory>


struct SimpleNet {
    std::vector<std::unique_ptr<AbstractSimpleLayer>> layers;
    
    SimpleNet() = default;

    void addLayer(std::unique_ptr<AbstractSimpleLayer> newLayer) {
        layers.push_back(std::move(newLayer));
    }

    std::vector<float> predict(const std::vector<float>& input) {

        std::vector<float> output = input;
        
        for(std::unique_ptr<AbstractSimpleLayer>& l : layers) {
            output = l->forward(output);
        }

        return output;
    }

    // don't need now, maybe later.
    void train() {}
};