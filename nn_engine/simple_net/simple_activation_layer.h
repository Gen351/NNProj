#pragma once

#include "abstract_simple_layer.h"

#include<cmath>
#include<vector>

class SimpleActivationLayer : public AbstractSimpleLayer {
    enum ActivationType {
        SIGM,
        RELU,
        LEAK,
        TANH
    };

    ActivationType activationType;

public:
    SimpleActivationLayer() : activationType(ActivationType::SIGM) {}
    SimpleActivationLayer(ActivationType act) : activationType(act) {}

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

    
	void save(std::ofstream& file) {

	}

    void load(std::ofstream& file) {

	}
    
	std::string getType() {
		return "ACT";
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