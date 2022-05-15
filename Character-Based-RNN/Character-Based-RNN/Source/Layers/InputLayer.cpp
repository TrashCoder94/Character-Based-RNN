#include "InputLayer.h"

InputLayer::InputLayer() : InputLayer(5)
{}

InputLayer::InputLayer(const size_t numberOfNeurons) : Layer(numberOfNeurons)
{}

InputLayer::~InputLayer()
{}

void InputLayer::InitLayer()
{
	std::vector<float> weightsIn{};
	weightsIn.reserve(GetNumberOfNeurons());

	for (size_t iN = 0; iN < GetNumberOfNeurons(); ++iN)
	{
		Neuron neuron{ weightsIn };
		const float newWeightIn = neuron.Init();
		weightsIn.push_back(newWeightIn);
		AddNeuron(neuron);
	}
}