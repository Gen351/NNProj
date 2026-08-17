#pragma once

#include "./abstract_simple_layer.h"

#include<cmath>
#include<vector>


class SimpleActivationLayer : public AbstractSimpleLayer {
public:
    enum ActivationType {
        SIGM,
        RELU,
        LEAK,
        TANH
    };

    SimpleActivationLayer() : activationType(ActivationType::SIGM) {}
    SimpleActivationLayer(ActivationType act) : activationType(act) {}

    ActivationType getActivation() const { return activationType; }
    void setActivation(ActivationType act) { activationType = act; }

    std::unique_ptr<AbstractSimpleLayer> clone() const override {
        return std::make_unique<SimpleActivationLayer>(*this);
    }

	std::string getType() const override {return "ACT";}

    std::string getActType() const {
        switch(getActivation()) {
            case SimpleActivationLayer::ActivationType::SIGM:
                return "SIGM";
                break;
            case SimpleActivationLayer::ActivationType::TANH:
                return "TANH";
                break;
            case SimpleActivationLayer::ActivationType::LEAK:
                return "LEAK";
                break;
            case SimpleActivationLayer::ActivationType::RELU:
                return "RELU";
                break;
        }
        return "SIGM";
    }
    

private:
    ActivationType activationType;

    std::vector<float> forward(const std::vector<float>& previousInput) override {
        std::vector<float> output(previousInput.size(), 0.0f);
        switch(activationType) {
            case SIGM:
                for(size_t i = 0; i < previousInput.size(); i++)
                    output[i] = Sigmoid(previousInput[i]);
                break;
            case RELU:
                for(size_t i = 0; i < previousInput.size(); i++)
                    output[i] = ReLU(previousInput[i]);
                break;
            case LEAK:
                for(size_t i = 0; i < previousInput.size(); i++)
                    output[i] = LeakyReLU(previousInput[i]);
                break;
            case TANH:
                for(size_t i = 0; i < previousInput.size(); i++)
                    output[i] = Tanh(previousInput[i]);
                break;
        }

        return output;
    }


private:
    float Sigmoid(float x) {
        return (1.0f / (1.0f + std::exp(-x)));
    }
    float ReLU(float x) {
        return (x > 0) * x;
    }
    float LeakyReLU(float x, float alpha=0.01) {
        return x * (x > 0) + (alpha * x) * (x <= 0);
    }
    float Tanh(float x) {
        return std::tanh(x);
    }

    // Doesn't need softmax 
    /*
        float SoftMax(std::vector<float> input, size_t out) {

        }
    */
};