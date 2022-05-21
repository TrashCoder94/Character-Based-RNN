#include "Layer.h"

Layer::Layer() : Layer(3)
{}

Layer::Layer(const size_t numberOfNeurons) :
	m_weights{},
	m_neurons{},
	m_numberOfNeurons{ numberOfNeurons }
{
	m_neurons.reserve(m_numberOfNeurons);
}

Layer::~Layer()
{
	m_weights.clear();
	m_neurons.clear();
}

const void Layer::PrintLayer() const
{
	std::cout << "==============" << std::endl;
	std::cout << "Printing Layer" << std::endl;
	std::cout << "==============" << std::endl;

	for (size_t iN = 0; iN < m_neurons.size(); ++iN)
	{
		const Neuron& neuron = m_neurons[iN];

		std::cout << "--------------" << std::endl;
		std::cout << "Neuron #" << iN << std::endl;

		neuron.Print();

		std::cout << "--------------" << std::endl;
	}

	std::cout << "==============" << std::endl;
	std::cout << "End of Layer" << std::endl;
	std::cout << "==============" << std::endl;
}

const float Layer::FeedForward(const std::vector<float> inputs) const
{
	float sumOfNeurons{ 0.0f };
	for (const Neuron& neuron : m_neurons)
		sumOfNeurons += neuron.FeedForward(inputs);
	return sumOfNeurons;
}

const std::vector<float> Layer::Recurrent(const std::vector<float> inputs) const
{
	std::vector<float> recurrent{};
	for (const Neuron& neuron : m_neurons)
		recurrent = neuron.Recurrent(inputs);
	return recurrent;
}
