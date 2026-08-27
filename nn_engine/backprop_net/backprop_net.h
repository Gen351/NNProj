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


namespace BackpropNet {

    class AbstractLayer {
    protected:
        bool training=false;
    public:
        virtual ~AbstractLayer() = default;

        virtual std::vector<float> forward(const std::vector<float>& input) = 0;
        virtual std::vector<float> backward(const std::vector<float>& output_gradient) = 0;
        virtual void update_weights_and_bias(float learning_rate) = 0;

        virtual size_t get_output_size() = 0;

        virtual std::string get_type() const = 0;

        void predict() {training = false;}
        void train() {training = true;}
    };

    struct Net {
        std::vector<std::unique_ptr<AbstractLayer>> layers;

        void add_layer(std::unique_ptr<AbstractLayer> new_layer) {
            layers.push_back(std::move(new_layer));
        }
 

        void train(size_t epochs
                , size_t batch_size
                , size_t patience
                , float learning_rate
                , const std::vector<std::vector<float>>& inputs
                , const std::vector<std::vector<float>>& target_values)
        {    
            // Dimension Checks
            if(inputs.empty() || inputs.size() != target_values.size()) {
                throw std::runtime_error("BackpropNet::Net.train(): inputs/target_values size mismatch or empty dataset!");
            }

            // Final Layer Dimension Check
            // DO a FORWARD PASS to update ActLayers' last_output
            for(auto& layer : layers) {layer->forward(inputs[0]);}
            if(!layers.empty()) {
                size_t final_layer_output = layers.back()->get_output_size();
                size_t expected_target_size = target_values[0].size();

                if(final_layer_output > 0 && final_layer_output != expected_target_size) {
                    throw std::runtime_error("BackpropNet::Net.train(): Final layer output dimension does not match target vector dimension!");
                }
            }


            // set the layers to training mode, so they update the weights n biases 
            for(auto& layer : layers) {layer->train();}

            // set previous loss to max loss
            float prev_loss = -1.0f;

            // 1. Display configs


            // 2. Main loop
            for(size_t epoch = 0; epoch < epochs; epoch++) {
                
                size_t batch = 0;
                
                for(size_t i = 0; i < inputs.size(); i++) {
                    
                    // 3. Foward Pass (Predict)
                    std::vector<float> predict_output = inputs[i];
                    for(auto& layer : layers) {
                        predict_output = layer->forward(predict_output);
                    }
                    
                    // 4. Calculate Loss Gradient
                    std::vector<float> loss_gradient(target_values[i].size());
                    for(size_t j = 0; j < target_values[i].size(); j++) {
                        loss_gradient[j] = (predict_output[j] - target_values[i][j]); 
                    }
                    
                    // 5. Backwards Pass
                    for(int j = static_cast<int>(layers.size()) - 1; j >= 0; j--) {
                        loss_gradient = layers[j]->backward(loss_gradient);
                    }

                    // 6. Batch updates
                    batch++;
                    if(batch == batch_size || i == inputs.size() - 1) {

                        // 7. Update the bias and weights
                        for(auto& layer : layers) {
                            layer->update_weights_and_bias(learning_rate / (float)batch);
                        }

                        batch = 0;
                    }
                }
            }
        }

        void train_v2(size_t epochs,
           size_t batch_size,
           size_t patience,
           float learning_rate,
           const std::vector<std::vector<float>>& inputs,
           const std::vector<std::vector<float>>& target_values,
           const std::vector<std::vector<float>>* val_inputs = nullptr,
           const std::vector<std::vector<float>>* val_targets = nullptr)
        {
            if(inputs.empty() || inputs.size() != target_values.size()) {
                throw std::runtime_error("Net::train(): Dataset mismatch or empty!");
            }

            for(auto& layer : layers) { layer->train(); }

            std::cout << "Training Setup {\n\tEpochs: [" << epochs 
                    << "]\n\tLearning Rate: [" << learning_rate 
                    << "]\n\tBatch Size: [" << batch_size 
                    << "]\n\tTraining Samples: [" << inputs.size() << "]\n}\n";

            float best_loss = -1.0f;
            size_t patience_counter = 0;
            const size_t MAX_PATIENCE = patience;

            std::vector<size_t> sample_order(inputs.size());
            std::iota(sample_order.begin(), sample_order.end(), 0);
            std::mt19937 rng(42); // Seeded for reproducibility

            for(size_t epoch = 0; epoch < epochs; epoch++) {
                std::time_t start = std::time(nullptr);
                std::cout << "Epoch \033[0;92m" << epoch + 1 << "/" << epochs << "\033[0m | Start: " << std::ctime(&start);

                std::shuffle(sample_order.begin(), sample_order.end(), rng);
                float total_loss = 0.0f;
                size_t batch_count = 0;

                for(size_t i = 0; i < inputs.size(); i++) {
                    size_t idx = sample_order[i];

                    // 1. Forward Pass
                    std::vector<float> predict_output = inputs[idx];
                    for(auto& layer : layers) {
                        predict_output = layer->forward(predict_output);
                    }

                    // 2. Mean Squared Error (MSE) Loss & Gradient (dL/dOutput)
                    // Works for raw linear outputs like Centipawns/Evaluation scores
                    std::vector<float> loss_gradient(target_values[idx].size());
                    for(size_t j = 0; j < target_values[idx].size(); j++) {
                        float diff = predict_output[j] - target_values[idx][j];
                        total_loss += diff * diff;
                        loss_gradient[j] = diff; // Gradient of 0.5 * (y_hat - y)^2
                    }

                    // 3. Backward Pass
                    for(int j = static_cast<int>(layers.size()) - 1; j >= 0; j--) {
                        loss_gradient = layers[j]->backward(loss_gradient);
                    }

                    // 4. Batch Updates (Adam)
                    batch_count++;
                    if(batch_count == batch_size || i == inputs.size() - 1) {
                        float gradient_scale = learning_rate / static_cast<float>(batch_count);
                        for(auto& layer : layers) {
                            layer->update_weights_and_bias(gradient_scale);
                        }
                        batch_count = 0;
                    }
                }

                float avg_loss = total_loss / static_cast<float>(inputs.size());
                printf("\tTrain MSE Loss=\033[0;92m%.7f\033[0m | Lrate=\033[0;92m%.8f\033[0m | ", avg_loss, learning_rate);
                std::time_t end = std::time(nullptr);
                std::cout << "Elapsed: " << (end - start) << "s\n";

                // 5. Validation Evaluation
                float current_eval_loss = avg_loss;

                if(val_inputs && val_targets && !val_inputs->empty()) {
                    for(auto& layer : layers) { layer->predict(); }
                    
                    float val_loss = 0.0f;
                    for(size_t k = 0; k < val_inputs->size(); k++) {
                        std::vector<float> out = (*val_inputs)[k];
                        for(auto& layer : layers) { out = layer->forward(out); }
                        
                        for(size_t j = 0; j < out.size(); j++) {
                            float diff = out[j] - (*val_targets)[k][j];
                            val_loss += diff * diff;
                        }
                    }
                    val_loss /= static_cast<float>(val_inputs->size());
                    printf("\tVal MSE Loss=\033[0;92m%.7f\033[0m\n", val_loss);
                    
                    current_eval_loss = val_loss;
                    for(auto& layer : layers) { layer->train(); }
                }

                // 6. Learning Rate Decay on Plateau
                if(best_loss < 0.0f || current_eval_loss < best_loss) {
                    best_loss = current_eval_loss;
                    patience_counter = 0;
                } else {
                    patience_counter++;
                    if(patience_counter >= MAX_PATIENCE) {
                        learning_rate *= 0.5f;
                        if(learning_rate < 1e-6f) learning_rate = 1e-6f;
                        printf("\t--- No improvement. Decay LR to %.8f ---\n", learning_rate);
                        patience_counter = 0;
                    }
                }
            }
        }


        std::vector<float> predict(const std::vector<float>& input) {
            // set the layers to predcit mode so they don't update weights
            for(auto& layer : layers) {layer->predict();}

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
        net.add_layer(std::make_unique<BackpropNet::DenseLayer>(781, 256, BackpropNet::DenseLayer::InitType::HE));
        net.add_layer(std::make_unique<BackpropNet::ActivationLayer>(BackpropNet::ActivationLayer::RELU));

        // Output layer for Linear Evaluation -> Use XAVIER initialization
        net.add_layer(std::make_unique<BackpropNet::DenseLayer>(256, 1, BackpropNet::DenseLayer::InitType::XAVIER));
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
        Matrix<float> weight_gradients;
        std::vector<float> biases;
        std::vector<float> bias_gradients;

        // Adam Optimizer state parameters
        Matrix<float> m_w; // 1st moment vector for weights
        Matrix<float> v_w; // 2nd moment vector for weights
        std::vector<float> m_b; // 1st moment vector for biases
        std::vector<float> v_b; // 2nd moment vector for biases
        size_t t = 0; // Step counter for Adam bias-correction

        std::vector<float> last_input;

    public:
        DenseLayer() = default;

        // Constructor with configurable weight initialization (defaults to HE)
        DenseLayer(size_t input, size_t output, InitType init = InitType::HE) {
            weights = Matrix<float>(output, input);
            weight_gradients = Matrix<float>(output, input);
            biases = std::vector<float>(output, 0.0f); // Biases initialized to 0
            bias_gradients = std::vector<float>(output, 0.0f);

            // Initialize Adam accumulators to 0
            m_w = Matrix<float>(output, input);
            v_w = Matrix<float>(output, input);
            m_b = std::vector<float>(output, 0.0f);
            v_b = std::vector<float>(output, 0.0f);

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
            if(training) { last_input = input; }

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

        std::vector<float> backward(const std::vector<float>& output_gradient) override {
            size_t output_size = output_gradient.size();
            size_t input_size = weights.cols();

            // 1. Accumulate bias gradients
            for(size_t i = 0; i < output_size; i++) {
                bias_gradients[i] += output_gradient[i];
            }

            // 2. Accumulate weight gradients
            for(size_t i = 0; i < output_size; i++) {
                for(size_t j = 0; j < input_size; j++) {
                    weight_gradients(i, j) += output_gradient[i] * last_input[j];
                }
            }

            // 3. Calculate input gradient for previous layer
            std::vector<float> input_gradient(input_size, 0.0f);
            for(size_t j = 0; j < input_size; j++) {
                float sum = 0.0f;
                for(size_t i = 0; i < output_size; i++) {
                    sum += weights(i, j) * output_gradient[i];
                }
                input_gradient[j] = sum;
            }

            return input_gradient;
        }

        void update_weights_and_bias(float learning_rate) override {
            t++; // Advance Adam timestep

            const float beta1 = 0.9f;
            const float beta2 = 0.999f;
            const float epsilon = 1e-8f;

            float beta1_t = std::pow(beta1, static_cast<float>(t));
            float beta2_t = std::pow(beta2, static_cast<float>(t));

            // Update Weights via Adam
            for(size_t i = 0; i < weights.rows(); i++) {
                for(size_t j = 0; j < weights.cols(); j++) {
                    float g = weight_gradients(i, j);

                    m_w(i, j) = beta1 * m_w(i, j) + (1.0f - beta1) * g;
                    v_w(i, j) = beta2 * v_w(i, j) + (1.0f - beta2) * (g * g);

                    float m_hat = m_w(i, j) / (1.0f - beta1_t);
                    float v_hat = v_w(i, j) / (1.0f - beta2_t);

                    weights(i, j) -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
                    weight_gradients(i, j) = 0.0f; // Reset accumulated gradient
                }
            }

            // Update Biases via Adam
            for(size_t i = 0; i < biases.size(); i++) {
                float g = bias_gradients[i];

                m_b[i] = beta1 * m_b[i] + (1.0f - beta1) * g;
                v_b[i] = beta2 * v_b[i] + (1.0f - beta2) * (g * g);

                float m_hat = m_b[i] / (1.0f - beta1_t);
                float v_hat = v_b[i] / (1.0f - beta2_t);

                biases[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
                bias_gradients[i] = 0.0f; // Reset accumulated gradient
            }
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

        std::vector<float> last_input;
        std::vector<float> last_output;
    
    public:
        
        ActivationLayer() : type(ActType::SIGM) {}
        ActivationLayer(ActType t) : type(t) {}

        std::vector<float> forward(const std::vector<float>& input) override {
            if(training) {last_input = input;}
            
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
            
            if(training) {last_output = output;}

            return output;
        }

        std::vector<float> backward(const std::vector<float>& output_gradient) override {
            size_t output_gradient_size = output_gradient.size();
            std::vector<float> input_gradient(output_gradient_size);

            switch(type) {
                case SIGM:
                    for(size_t i = 0; i < output_gradient_size; i++) {input_gradient[i] = output_gradient[i] * (last_output[i] * (1.0f - last_output[i]));}
                    break;
                case LEAK:
                    for(size_t i = 0; i < output_gradient_size; i++) {input_gradient[i] = output_gradient[i] * LeakyReLUDerivative(last_input[i]);}
                    break;
                case RELU:
                    for(size_t i = 0; i < output_gradient_size; i++) {input_gradient[i] = output_gradient[i] * ReLUDerivative(last_input[i]);}
                    break;
                case TANH:
                    for(size_t i = 0; i < output_gradient_size; i++) {input_gradient[i] = output_gradient[i] * (1.0f - (last_output[i] * last_output[i]));}
                    break;
                case SOFT:
                    input_gradient = SoftMaxBackward(last_output, output_gradient);
                    break;
            }

            return input_gradient;
        }

        void update_weights_and_bias(float learning_rate) override {
            // Do nothing
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

        size_t get_output_size() override {
            return last_output.size();
        }

        ActType get_activation() const { return type; }
        std::string get_type() const override { return "ACT"; }
    };
}
