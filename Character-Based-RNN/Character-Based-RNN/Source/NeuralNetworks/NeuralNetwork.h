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
		void Process();
		void Print();

	private:
		InputLayer m_inputLayer;
		std::vector<HiddenLayer> m_hiddenLayers;
		OutputLayer m_outputLayer;

		float m_finalResult;

		size_t m_numberOfHiddenNeurons;
		size_t m_numberOfHiddenLayers;
};