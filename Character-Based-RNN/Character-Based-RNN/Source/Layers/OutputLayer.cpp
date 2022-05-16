#include "OutputLayer.h"

OutputLayer::OutputLayer() : OutputLayer(5, {})
{}

OutputLayer::OutputLayer(const size_t numberOfNeurons, const std::vector<float> weightsIn) : Layer(numberOfNeurons),
	m_weightsIn{ weightsIn }
{}

OutputLayer::~OutputLayer()
{
	m_weightsIn.clear();
}

void OutputLayer::InitLayer()
{
	for (size_t iN = 0; iN < GetNumberOfNeurons(); ++iN)
	{
		Neuron neuron{ m_weightsIn };
		const float newWeight = neuron.Init();
		AddWeight(newWeight);
		AddNeuron(neuron);
	}
}