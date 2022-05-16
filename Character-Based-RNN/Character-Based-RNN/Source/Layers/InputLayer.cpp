#include "InputLayer.h"

InputLayer::InputLayer() : InputLayer(5)
{}

InputLayer::InputLayer(const size_t numberOfNeurons) : Layer(numberOfNeurons)
{}

InputLayer::~InputLayer()
{}

void InputLayer::InitLayer()
{	
	for (size_t iN = 0; iN < GetNumberOfNeurons(); ++iN)
	{
		Neuron neuron{ GetWeights() };
		const float newWeight = neuron.Init();
		AddWeight(newWeight);
		AddNeuron(neuron);
	}
}