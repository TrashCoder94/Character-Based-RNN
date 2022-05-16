#include "Neuron.h"
#include <iostream>
#include <stdlib.h>
#include <random>

Neuron::Neuron() : Neuron(std::vector<float>())
{}

Neuron::Neuron(const std::vector<float> weightsIn, const float bias /*= 0.0f*/) :
	m_weightsIn{ weightsIn },
	m_bias{ bias },
	m_outputValue{ 0.0f },
	m_error{ 0.0f }
{}

Neuron::~Neuron()
{
	m_weightsIn.clear();
}

float Neuron::Init()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);
	return dis(gen);
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

const float Neuron::Sigmoid(const float x) const
{
	float sigmoid{ 1.0f / (1.0f + std::exp(-x)) };
	return sigmoid;
}