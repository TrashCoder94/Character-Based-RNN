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
		const float FeedForward(const std::vector<float> inputs) const;
		const std::vector<float> Recurrent(const std::vector<float> inputs) const;

		inline const void AddNeuron(const Neuron& neuron) { m_neurons.push_back(neuron); }
		inline const void AddWeight(const float& weight) { m_weights.push_back(weight); }

		inline const std::vector<Neuron>& GetNeurons() { return m_neurons; }
		inline const size_t& GetNumberOfNeurons() const { return m_numberOfNeurons; }
		inline const std::vector<float>& GetWeights() const { return m_weights; }

	private:
		std::vector<float> m_weights;
		std::vector<Neuron> m_neurons;
		size_t m_numberOfNeurons;
};