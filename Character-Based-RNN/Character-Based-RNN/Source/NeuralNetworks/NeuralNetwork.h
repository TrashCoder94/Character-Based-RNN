#pragma once

#include "../Layers/InputLayer.h"
#include "../Layers/HiddenLayer.h"
#include "../Layers/OutputLayer.h"

class NeuralNetwork
{
	public:
		NeuralNetwork();
		explicit NeuralNetwork(const std::vector<float> input, const size_t numberOfInputNeurons, const size_t numberOfHiddenNeurons, const size_t numberOfOutputNeurons);
		~NeuralNetwork();

		void Init();
		const float FeedForward(const std::vector<float> inputs) const;
		const std::vector<float> Recurrent(const std::vector<float> inputs, size_t iterations = 1) const;
		void Print();

	private:
		InputLayer m_inputLayer;
		HiddenLayer m_hiddenLayer;
		OutputLayer m_outputLayer;

		std::vector<float> m_inputValues;
		std::vector<float> m_inputWeights;

		size_t m_numberOfHiddenNeurons;
};