#pragma once

#include <iostream>
#include <vector>
#include "../Neurons/Neuron.h"

class Layer
{
	protected:
		Layer();
		explicit Layer(const size_t numberOfNeurons);

	public:
		virtual ~Layer();

		virtual void InitLayer() = 0;
		virtual const void PrintLayer() const;
		const float GetSumOfNeuronWeights() const;

		inline const void AddNeuron(const Neuron& neuron) { m_neurons.push_back(neuron); }
		inline const std::vector<Neuron>& GetNeurons() { return m_neurons; }
		inline const size_t& GetNumberOfNeurons() const { return m_numberOfNeurons; }

	private:
		std::vector<Neuron> m_neurons;
		size_t m_numberOfNeurons;
};