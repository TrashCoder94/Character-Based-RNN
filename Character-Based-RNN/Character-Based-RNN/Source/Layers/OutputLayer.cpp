#include "OutputLayer.h"

OutputLayer::OutputLayer() : OutputLayer(5, {}, {})
{}

OutputLayer::OutputLayer(const size_t numberOfNeurons, const std::vector<float> inputValues, const std::vector<float>& weightsIn) : Layer(numberOfNeurons),
	m_inputValues{ inputValues },
	m_weightsIn{ weightsIn }
{}

OutputLayer::~OutputLayer()
{
	m_inputValues.clear();
	m_weightsIn.clear();
}

void OutputLayer::InitLayer()
{
	for (size_t iN = 0; iN < GetNumberOfNeurons(); ++iN)
	{
		Neuron neuron{ m_inputValues, m_weightsIn };
		neuron.Init();
		AddNeuron(neuron);
	}
}