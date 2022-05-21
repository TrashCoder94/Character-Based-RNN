#include "HiddenLayer.h"

HiddenLayer::HiddenLayer() : HiddenLayer(5, {}, {})
{}

HiddenLayer::HiddenLayer(const size_t numberOfNeurons, const std::vector<float> inputValues, const std::vector<float> weightsIn) : Layer(numberOfNeurons),
	m_inputValues{ inputValues },
	m_weightsIn{ weightsIn }
{}

HiddenLayer::~HiddenLayer()
{
	m_inputValues.clear();
	m_weightsIn.clear();
}

void HiddenLayer::InitLayer()
{
	for (size_t iN = 0; iN < GetNumberOfNeurons(); ++iN)
	{
		Neuron neuron{ m_inputValues, m_weightsIn };
		neuron.Init();
		AddNeuron(neuron);
	}
}