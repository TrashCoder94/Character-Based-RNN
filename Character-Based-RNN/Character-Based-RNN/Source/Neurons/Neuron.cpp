#include "Neuron.h"
#include <iostream>
#include <stdlib.h>
#include <random>

Neuron::Neuron() : Neuron(std::vector<float>(), std::vector<float>())
{}

Neuron::Neuron(const std::vector<float> inputValues, const std::vector<float>& weightsIn, const float bias /*= 0.0f*/) :
	m_inputValues{ inputValues },
	m_weightsIn{ weightsIn },
	m_weight{ 0.0f },
	m_bias{ bias },
	m_outputValue{ 0.0f },
	m_error{ 0.0f }
{}

Neuron::~Neuron()
{
	m_inputValues.clear();
	m_weightsIn.clear();
}

float Neuron::Init()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);
	m_weight = dis(gen);
	return m_weight;
}

const void Neuron::Print() const
{
	std::cout << "Weights In:" << std::endl;
	for (size_t iW = 0; iW < m_weightsIn.size(); ++iW)
		std::cout << "W" << iW << " = " << m_weightsIn[iW] << std::endl;

	std::cout << "Bias = " << m_bias << std::endl;
	std::cout << "Error = " << m_error << std::endl;
}

const float Neuron::FeedForward(const std::vector<float> inputs) const
{
	float sum{ 0.0f };
	for (size_t iN = 0; iN < inputs.size(); ++iN)
	{
		const float input = inputs[iN];
		const float weight = m_weightsIn[iN];
		const float inputWeight = input * weight;
		sum += inputWeight;
	}
	sum = Sigmoid(sum) + m_bias;
	return sum;
}

const std::vector<float> Neuron::Recurrent(const std::vector<float> inputs) const
{
	std::vector<float> recurrent{};
	for (size_t iN = 0; iN < m_inputValues.size(); ++iN)
	{
		// h{t} = f(w_{x} x_t + w_{h} h_{t-1} + b_h)
		/*
			h{t} = current hidden state
			f = Sigmoid(value)
			w_x = m_weightsIn[iN]?
			x_t = m_inputLayerValues[iN] (Pass std::vector<float> in constructor to represent inputs)
			w_h = create new float variable m_weight and set this in Init()
			h_t-1 = inputs[iN]
		*/

		float currentHiddenState = Sigmoid((m_weightsIn[iN] * m_inputValues[iN]) + (m_weight * inputs[iN])) + m_bias;
		recurrent.push_back(currentHiddenState);
	}
	return recurrent;
}

const float Neuron::Sigmoid(const float x) const
{
	float sigmoid{ 1.0f / (1.0f + std::exp(-x)) };
	return sigmoid;
}