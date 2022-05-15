#include "Neuron.h"
#include <iostream>
#include <stdlib.h>

Neuron::Neuron() : Neuron(std::vector<float>())
{}

Neuron::Neuron(const std::vector<float> weightsIn) :
	m_weightsIn{ weightsIn },
	m_outputValue{ 0.0f },
	m_error{ 0.0f }
{}

Neuron::~Neuron()
{}

float Neuron::Init()
{
	return (((float)rand()) / (float)RAND_MAX);
}

const void Neuron::Print() const
{
	std::cout << "Weights In:" << std::endl;
	for (size_t iW = 0; iW < m_weightsIn.size(); ++iW)
		std::cout << "W" << iW << " = " << m_weightsIn[iW] << std::endl;

	std::cout << "Error = " << m_error << std::endl;
	std::cout << "Output Value = " << GetSum() << std::endl;
}

const float Neuron::GetSum() const
{
	float sum{ 0.0f };
	for (float weightIn : m_weightsIn)
		sum += weightIn;
	return sum;
}