#pragma once

#include <vector>

class Neuron
{
	public:
		Neuron();
		explicit Neuron(const std::vector<float> inputValues, const std::vector<float>& weightsIn, const float bias = 0.0f);
		~Neuron();

		float Init();
		const void Print() const;
		const float FeedForward(const std::vector<float> inputs) const;
		const std::vector<float> Recurrent(const std::vector<float> inputs) const;
		const float Sigmoid(const float x) const;

	private:
		std::vector<float> m_inputValues;
		std::vector<float> m_weightsIn;
		float m_weight;
		float m_bias;
		float m_outputValue;
		float m_error;
};