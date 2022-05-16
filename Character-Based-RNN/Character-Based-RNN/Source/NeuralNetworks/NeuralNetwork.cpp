#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() : NeuralNetwork(5, 5, 3, 1)
{}

NeuralNetwork::NeuralNetwork(const size_t numberOfInputNeurons, const size_t numberOfHiddenNeurons, const size_t numberOfHiddenLayers, const size_t numberOfOutputNeurons) :
	m_inputLayer{ numberOfInputNeurons },
	m_hiddenLayers{},
	m_outputLayer{ numberOfOutputNeurons, {} },
	m_inputWeights{},
	m_numberOfHiddenNeurons{ numberOfHiddenNeurons },
	m_numberOfHiddenLayers{ numberOfHiddenLayers }
{}

NeuralNetwork::~NeuralNetwork()
{}

void NeuralNetwork::Init()
{
	//
	// INPUT LAYER
	//

	m_inputLayer.InitLayer();
	m_inputWeights = m_inputLayer.GetWeights();

	// 
	// HIDDEN LAYER
	//

	std::vector<float> previousHiddenLayerWeights{ m_inputWeights };
	for (size_t iH = 0; iH < m_numberOfHiddenLayers; ++iH)
	{
		HiddenLayer hiddenLayer{ m_numberOfHiddenNeurons, previousHiddenLayerWeights };
		hiddenLayer.InitLayer();
		m_hiddenLayers.push_back(hiddenLayer);
		if(m_numberOfHiddenLayers > 1)
			previousHiddenLayerWeights = m_hiddenLayers[iH - 1].GetWeights();
	}

	// OUTPUT LAYER
	m_outputLayer = OutputLayer(1, previousHiddenLayerWeights);
	m_outputLayer.InitLayer();
}

const float NeuralNetwork::FeedForward(std::vector<float> inputs) const
{
	std::vector<float> feedForwardResults(m_numberOfHiddenLayers, 0.0f);
	for (const HiddenLayer& hiddenLayer : m_hiddenLayers)
	{
		const float feedForward = hiddenLayer.FeedForward(inputs);
		feedForwardResults.push_back(feedForward);
	}

	const float feedForwardFinalResult = m_outputLayer.FeedForward(feedForwardResults);
	return feedForwardFinalResult;
}

void NeuralNetwork::Print()
{
	std::cout << "==============" << std::endl;
	std::cout << "INPUT LAYER" << std::endl;
	std::cout << "==============" << std::endl;
	
	m_inputLayer.PrintLayer();

	std::cout << "==============" << std::endl;
	std::cout << "END OF INPUT LAYER" << std::endl;
	std::cout << "==============" << std::endl;

	std::cout << "==============" << std::endl;
	std::cout << "HIDDEN LAYERS" << std::endl;
	std::cout << "==============" << std::endl;

	for (size_t iH = 0; iH < m_numberOfHiddenLayers; ++iH)
	{
		std::cout << "--------------" << std::endl;
		std::cout << "Hidden Layer #" << iH << std::endl;

		m_hiddenLayers[iH].PrintLayer();
		
		std::cout << "--------------" << std::endl;
	}

	std::cout << "==============" << std::endl;
	std::cout << "END OF HIDDEN LAYERS" << std::endl;
	std::cout << "==============" << std::endl;

	std::cout << "==============" << std::endl;
	std::cout << "OUTPUT LAYER" << std::endl;
	std::cout << "==============" << std::endl;

	m_outputLayer.PrintLayer();

	std::cout << "==============" << std::endl;
	std::cout << "END OF OUTPUT LAYER" << std::endl;
	std::cout << "==============" << std::endl;
}