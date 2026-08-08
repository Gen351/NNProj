#pragma once

#include "../matrix_op/matrix.hpp"
#include "abstract_simple_layer.h"

#include<vector>
#include<cmath>
#include<random>
#include<stdexcept>

class SimpleLayer : public AbstractSimpleLayer {
	size_t input;
	size_t output;

public:
	// @brief Rows=output, Cols=input
	Matrix<float> weights;
	std::vector<float> biases;

	SimpleLayer() : weights(1, 1)
					, biases(1, 0.0f)
					, input(1)
					, output(1)	
	{
		// Init random weights
		for(float& w : weights.data) w = initWeightHeNormal(1);
	}

	SimpleLayer(size_t in, size_t out) : weights(out, in)
										, biases(out, 0.0f)
										, input(in)
										, output(out) 
	{
		// Init random weights
		for(float& w : weights.data) w = initWeightHeNormal(in);
	}

	std::vector<float> forward(const std::vector<float>& previousInput) override {
		#if NDEBUG
			if(previousInput.size() != input)
				throw std::runtime_error("Forward: Input Misallignment [previousInput.size() != input]");
		#endif

		std::vector<float> output;
		
		for(size_t i = 0; i < weights.rows(); i++){
			float sum = biases[i];
			for(size_t j = 0; j < weights.cols(); j++) { 
				sum += (previousInput[j] * weights(i, j));
			}
			output[i] = sum;
		}
		return output;
	}

	void save(std::ofstream& file) {

	}

    void load(std::ofstream& file) {

	}
    
	std::string getType() {
		return "LAY";
	}


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
