#pragma once

#include "../matrix_op/matrix.hpp"
#include "./abstract_simple_layer.h"

#include<vector>
#include<cmath>
#include<random>
#include<stdexcept>

class SimpleLayer : public AbstractSimpleLayer {
	size_t input_size;
	size_t output_size;

public:
	// @brief Rows=output_size, Cols=input_size
	Matrix<float> weights;
	std::vector<float> biases;

	SimpleLayer() : weights(1, 1)
					, biases(1, 0.0f)
					, input_size(1)
					, output_size(1)	
	{
		// Init random weights
		for(float& w : weights.data) w = initWeightHeNormal(1);
	}

	SimpleLayer(size_t in, size_t out) : weights(out, in)
										, biases(out, 0.0f)
										, input_size(in)
										, output_size(out) 
	{
		// Init random weights
		for(float& w : weights.data) w = initWeightHeNormal(in);
	}

	std::vector<float> forward(const std::vector<float>& previousInput) override {
		#ifndef NDEBUG
			if(previousInput.size() != input_size)
				throw std::runtime_error("Forward: Input Misallignment [previousInput.size() != input_size]");
		#endif

		std::vector<float> output(output_size);
		
		for(size_t i = 0; i < weights.rows(); i++){
			float sum = biases[i];
			for(size_t j = 0; j < weights.cols(); j++) { 
				sum += (previousInput[j] * weights(i, j));
			}
			output[i] = sum;
		}
		return output;
	}

	std::string getType() const override {return "LAY";}

private:
	// Not by me
	float initWeightHeNormal(int fanIn) {
		static thread_local std::mt19937 engine(std::random_device{}());
		
		float stddev = std::sqrt(2.0f / static_cast<float>(fanIn));
		std::normal_distribution<float> dist(0.0f, stddev);
		
		return dist(engine);
	}
	// Not by me
	float initWeightHeUniform(int fanIn) {
		static thread_local std::mt19937 engine(std::random_device{}());
		
		float limit = std::sqrt(6.0f / static_cast<float>(fanIn));
		std::uniform_real_distribution<float> dist(-limit, limit);
		
		return dist(engine);
	}
};
