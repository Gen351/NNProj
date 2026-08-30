#pragma once

#include<iostream>
#include<random>
#include<ctime>
#include<algorithm>
#include<numeric>
#include<cstdio>

#include<vector>
#include<string>
#include<memory>
#include<cmath>

#include "../matrix_op/matrix.hpp"


namespace NNUE {

    class AbstractLayer {
    public:
        virtual ~AbstractLayer() = default;

        virtual std::vector<float> forward(const std::vector<float>& input) = 0;
        virtual void update_weights_and_bias(float learning_rate) = 0;

        virtual size_t get_output_size() = 0;

        virtual std::string get_type() const = 0;
    };

    struct Net {
        std::vector<std::unique_ptr<AbstractLayer>> layers;

        void add_layer(std::unique_ptr<AbstractLayer> new_layer) {
            layers.push_back(std::move(new_layer));
        }

        std::vector<float> predict(const std::vector<float>& input) {

            std::vector<float> output = input;
            for(const auto& layer : layers) {
                output = layer->forward(output);
            }

            return output;
        }
    };




    
    /** Usage Example:
     * 
        //Hidden layers with ReLU -> Use HE initialization (default)
        net.add_layer(std::make_unique<NNUE::DenseLayer>(781, 256, NNUE::DenseLayer::InitType::HE));
        net.add_layer(std::make_unique<NNUE::ActivationLayer>(NNUE::ActivationLayer::RELU));

        // Output layer for Linear Evaluation -> Use XAVIER initialization
        net.add_layer(std::make_unique<NNUE::DenseLayer>(256, 1, NNUE::DenseLayer::InitType::XAVIER));
     */
    class DenseLayer : public AbstractLayer {
    public:
        enum class InitType {
            HE,      // He / Kaiming Normal (Best for ReLU / LeakyReLU)
            XAVIER,  // Xavier / Glorot Uniform (Best for Sigmoid / Tanh / Linear)
            RANDOM   // Small Uniform Random [-0.05, 0.05]
        };

    private:
        Matrix<float> weights;
        std::vector<float> biases;

    public:
        DenseLayer() = default;

        // Constructor with configurable weight initialization (defaults to HE)
        DenseLayer(size_t input, size_t output, InitType init = InitType::HE) {
            weights = Matrix<float>(output, input);
            biases = std::vector<float>(output, 0.0f); // Biases initialized to 0

            initialize_weights(input, output, init);
        }

    private:
        void initialize_weights(size_t fan_in, size_t fan_out, InitType init) {
            std::mt19937 rng(std::random_device{}());

            if (init == InitType::HE) {
                // He / Kaiming Normal: Normal distribution N(0, sqrt(2 / fan_in))
                float stddev = std::sqrt(2.0f / static_cast<float>(fan_in));
                std::normal_distribution<float> dist(0.0f, stddev);
                for (size_t i = 0; i < weights.rows(); i++) {
                    for (size_t j = 0; j < weights.cols(); j++) {
                        weights(i, j) = dist(rng);
                    }
                }
            } 
            else if (init == InitType::XAVIER) {
                // Xavier / Glorot Uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
                float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
                std::uniform_real_distribution<float> dist(-limit, limit);
                for (size_t i = 0; i < weights.rows(); i++) {
                    for (size_t j = 0; j < weights.cols(); j++) {
                        weights(i, j) = dist(rng);
                    }
                }
            } 
            else { // RANDOM
                std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
                for (size_t i = 0; i < weights.rows(); i++) {
                    for (size_t j = 0; j < weights.cols(); j++) {
                        weights(i, j) = dist(rng);
                    }
                }
            }
        }

    public:
        std::vector<float> forward(const std::vector<float>& input) override {
            std::vector<float> output(biases.size());

            for(size_t i = 0; i < weights.rows(); i++) {
                float sum = biases[i];
                for(size_t j = 0; j < weights.cols(); j++) {
                    sum += input[j] * weights(i, j);
                }
                output[i] = sum;
            }

            return output;
        }
 
        size_t get_output_size() override {
            return biases.size();
        }

        std::string get_type() const override { return "DEN"; }

        const Matrix<float>& get_weights() const { return weights; }
        Matrix<float>& get_weights() { return weights; }
        const std::vector<float>& get_biases() const { return biases; }
        std::vector<float>& get_biases() { return biases; }
    };
    
    
    
    class ActivationLayer : public AbstractLayer {
    public:
        enum ActType {
            SIGM,
            LEAK,
            RELU,
            TANH,
            SOFT,
        };

    private:
        ActType type;

    public:
        
        ActivationLayer() : type(ActType::SIGM) {}
        ActivationLayer(ActType t) : type(t) {}

        std::vector<float> forward(const std::vector<float>& input) override {
            
            size_t size = input.size();
            std::vector<float> output(size);

            switch(type) {
                case SIGM:
                    for(size_t i = 0; i < size; i++) {output[i] = Sigmoid(input[i]);}
                    break;
                case LEAK:
                    for(size_t i = 0; i < size; i++) {output[i] = LeakyReLU(input[i]);}
                    break;
                case RELU:
                    for(size_t i = 0; i < size; i++) {output[i] = ReLU(input[i]);}
                    break;
                case TANH:    
                    for(size_t i = 0; i < size; i++) {output[i] = Tanh(input[i]);}
                    break;
                case SOFT:
                    output = SoftMax(input);
                    break;
            }

            return output;
        }
        
        inline static float Sigmoid(float x) {
            return (1 / (1 + std::exp(-x)));
        }

        inline static float LeakyReLU(float x, float alpha=0.01) {
            return x * (x > 0) + (alpha * x) * (x <= 0);
        }
        inline static float LeakyReLUDerivative(float x, float alpha=0.01) {
            return (x > 0) ? 1.0f : alpha;
        }

        inline static float ReLU(float x) {
            return (x > 0) * x;
        }
        inline static float ReLUDerivative(float x) {
            return float(x > 0);
        }

        inline static float Tanh(float x) {
            return std::tanh(x);
        }

        inline static std::vector<float> SoftMax(const std::vector<float>& input) {
            const size_t size = input.size();
            std::vector<float> output(size);
            float max = -__FLT_MAX__;
            // Get max
            for(size_t i = 0; i < size; i++) {
                if(max < input[i]) {
                    max = input[i];
                }
            }

            float sum = 0.0f;
            // Get sum
            // SoftMax(Z) = x_i / sum(exp(Z=input_i - max))
            for(size_t i = 0; i < size; i++) {
                output[i] = std::exp(input[i] - max);
                sum += output[i];
            }

            for(size_t i = 0; i < size; i++) {
                output[i] /= sum;
            }

            return output;
        }
        inline static std::vector<float> SoftMaxBackward(const std::vector<float>& output, const std::vector<float>& output_gradient) {
            size_t size = output.size();
            std::vector<float> input_gradient(size, 0.0f);

            // 1. Calculate the dot product of upstream gradients and softmax outputs:
            // sum(dL/da_i * a_i)
            float dot_product = 0.0f;
            for (size_t i = 0; i < size; ++i) {
                dot_product += output_gradient[i] * output[i];
            }

            // 2. Compute the final gradient for each input logit:
            // dL/dz_j = a_j * (dL/da_j - dot_product)
            for (size_t j = 0; j < size; ++j) {
                input_gradient[j] = output[j] * (output_gradient[j] - dot_product);
            }

            return input_gradient;
        }

        ActType get_activation() const { return type; }
        std::string get_type() const override { return "ACT"; }
    };
}
