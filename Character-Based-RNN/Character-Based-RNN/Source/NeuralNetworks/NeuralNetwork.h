#pragma once

#include "../Layers/InputLayer.h"
#include "../Layers/HiddenLayer.h"
#include "../Layers/OutputLayer.h"

class NeuralNetwork
{
	public:
		NeuralNetwork();
		explicit NeuralNetwork(const size_t numberOfInputNeurons, const size_t numberOfHiddenNeurons, const size_t numberOfHiddenLayers, const size_t numberOfOutputNeurons);
		~NeuralNetwork();

		void Init();
		const float FeedForward(std::vector<float> inputs) const;
		void Print();

	private:
		InputLayer m_inputLayer;
		std::vector<HiddenLayer> m_hiddenLayers;
		OutputLayer m_outputLayer;

		std::vector<float> m_inputWeights;

		size_t m_numberOfHiddenNeurons;
		size_t m_numberOfHiddenLayers;
};