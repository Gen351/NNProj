#pragma once

#include<vector>
#include<memory>
#include<cmath>

#include "../matrix_op/matrix.hpp"



namespace BackpropNet {
    
    struct Net {
        std::vector<std::unique_ptr<AbstractLayer>> layers;


        void add_layer(AbstractLayer* new_layer) {
            layers.push_back(std::unique_ptr<AbstractLayer>(new_layer));
        }
 

        void train(size_t epochs
                , float learning_rate
                , const Matrix<float>& inputs
                , const Matrix<float>& target_values) {
            
            // set the layers to training mode, so the update the weights 
            for(auto& layer : layers) {layer->train();}
            


        }


        std::vector<float> predict(const std::vector<float>& input) {
            // set the layers to predcit mode so they don't update weights
            for(auto& layer : layers) {layer->predict();}



            return std::vector<float>();
        }
    };





    class AbstractLayer {
    protected:
        bool training=false;    
    public:
        virtual ~AbstractLayer() = default;

        virtual std::vector<float> forward(const std::vector<float>& input) = 0;
        virtual std::vector<float> backward(const std::vector<float>& output_gradient) = 0;
        virtual void update_weights(float learning_rate) = 0;
        
        void predict() {training = false;}
        void train() {training = true;}
    };
    
    
    
    
    
    class DenseLayer : public AbstractLayer {
        Matrix<float> weights;
        Matrix<float> weight_gradients;
        std::vector<float> biases;
        std::vector<float> bias_gradients;

        std::vector<float> last_input;
    public:

        DenseLayer() = default;
        DenseLayer(size_t input, size_t output) {
            weights = Matrix<float>(output, input);
            weight_gradients = Matrix<float>(output, input);
            biases = std::vector<float>(output);
            bias_gradients = std::vector<float>(output);
            
            last_input = std::vector<float>(input);
        }    
        

        std::vector<float> forward(const std::vector<float>& input) override {
            if(training) {last_input = input;}

            std::vector<float> output(biases.size());

            for(int i = 0; i < weights.rows(); i++) {
                float sum = biases[i];
                for(int j = 0; j < weights.cols(); j++) {
                    sum += (input[j]) * weights(i, j);
                }
                output[i] = sum;
            }

            return output;
        }

        std::vector<float> backward(const std::vector<float>& output_gradient) override {

        }

        void update_weights(float learning_rate) override {

        }  
    };
    
    
    
    
    
    class ActivationLayer : public AbstractLayer {
        enum ActType {
            SIGM,
            LEAK,
            RELU,
            TANH,
            SOFT,
        };

        ActType type;

        std::vector<float> last_input;
    public:
        
        ActivationLayer() {
            type = ActType::SIGM;
            last_input = std::vector<float>();
        }
        ActivationLayer(ActType t) {
            type = t;
            last_input = std::vector<float>();
        }


        std::vector<float> forward(const std::vector<float>& input) override {
            if(training) {last_input = input;}

            std::vector<float> output(input.size());

            switch(type) {
                case SIGM:
                    for(int i = 0; i < output.size(); i++) {output[i] = Sigmoid(input[i]);}
                    break;
                case LEAK:
                    for(int i = 0; i < output.size(); i++) {output[i] = LeakyReLU(input[i]);}
                    break;
                case RELU:
                    for(int i = 0; i < output.size(); i++) {output[i] = ReLU(input[i]);}
                    break;
                case TANH:    
                    for(int i = 0; i < output.size(); i++) {output[i] = Tanh(input[i]);}
                    break;
                case SOFT:
                    output = SoftMax(input);
                    break;
            }

            return output;
        }

        std::vector<float> backward(const std::vector<float>& output_gradient) override {

        }

        void update_weights(float learning_rate) override {

        }
        
    private:
        static float Sigmoid(float x) {
            return (1 / (1 + std::exp(-x)));
        }
        static float SigmoidDerivative(const float& x) {
            const float sigmoid_derivative = Sigmoid(x);
            return sigmoid_derivative * (1.0f - sigmoid_derivative);
        }

        static float LeakyReLU(float x, float alpha=0.01) {
            return x * (x > 0) + (alpha * x) * (x <= 0);
        }
        static float LeakyReLUDerivative(float x, float alpha=0.01) {
            return 1.0f * (x > 0) + (alpha * x) * (x <= 0);
        }

        static float ReLU(float x) {
            return (x > 0) * x;
        }
        static float ReLUDerivative(float x) {
            return float(x > 0);
        }

        static float Tanh(float x) {
            return std::tanh(x);
        }
        static float TanhDerivative(const float& x) {
            const float tanh_x = std::tanh(x);
            return 1.0f - (tanh_x * tanh_x);
        }

        static std::vector<float> SoftMax(const std::vector<float>& input) {
            const size_t size = input.size();
            std::vector<float> output(size);
            float max = -__FLT_MAX__;
            // Get max
            for(int i = 0; i < size; i++) {
                if(max < input[i]) {
                    max = input[i];
                }
            }

            float sum = 0.0f;
            // Get sum
            // SoftMax(Z) = x_i / sum(exp(Z=input_i - max))
            for(int i = 0; i < size; i++) {
                output[i] = std::exp(input[i] - max);
                sum += output[i];
            }

            for(int i = 0; i < size; i++) {
                output[i] /= sum;
            }

            return output;
        }
        static std::vector<float> SoftMaxBackward(const std::vector<float>& output, const std::vector<float>& output_gradient) {
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

    };
    
    






}
