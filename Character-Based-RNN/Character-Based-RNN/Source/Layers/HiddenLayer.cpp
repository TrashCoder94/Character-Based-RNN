#include "HiddenLayer.h"

HiddenLayer::HiddenLayer() : HiddenLayer(5, {})
{}

HiddenLayer::HiddenLayer(const size_t numberOfNeurons, const std::vector<float> weightsIn) : Layer(numberOfNeurons),
	m_weightsIn{ weightsIn },
	m_weightsOut{}
{}

HiddenLayer::~HiddenLayer()
{
	m_weightsIn.clear();
	m_weightsOut.clear();
}

void HiddenLayer::InitLayer()
{
	for (size_t iN = 0; iN < GetNumberOfNeurons(); ++iN)
	{
		Neuron neuron{ m_weightsIn };
		const float newWeightOut = neuron.Init();
		m_weightsOut.push_back(newWeightOut);
		AddNeuron(neuron);
	}
}