#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() : NeuralNetwork({2.0f, 3.0f}, 2, 2, 1)
{}

NeuralNetwork::NeuralNetwork(const std::vector<float> input, const size_t numberOfInputNeurons, const size_t numberOfHiddenNeurons, const size_t numberOfOutputNeurons) :
	m_inputLayer{ numberOfInputNeurons },
	m_hiddenLayer{ numberOfHiddenNeurons, {}, {} },
	m_outputLayer{ numberOfOutputNeurons, {}, {} },
	m_inputValues{ input },
	m_inputWeights{},
	m_numberOfHiddenNeurons{ numberOfHiddenNeurons }
{}

NeuralNetwork::~NeuralNetwork()
{
	m_inputValues.clear();
	m_inputWeights.clear();
}

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
	m_hiddenLayer = HiddenLayer{ m_numberOfHiddenNeurons, m_inputValues, m_inputWeights };
	m_hiddenLayer.InitLayer();

	//
	// OUTPUT LAYER
	//
	m_outputLayer = OutputLayer{ 1, m_inputValues, m_inputWeights };
	m_outputLayer.InitLayer();
}

const float NeuralNetwork::FeedForward(std::vector<float> inputs) const
{
#if 0
	std::vector<float> feedForwardResults(m_numberOfHiddenLayers, 0.0f);
	for (const HiddenLayer& hiddenLayer : m_hiddenLayers)
	{
		const float feedForward = hiddenLayer.FeedForward(inputs);
		feedForwardResults.push_back(feedForward);
	}

	const float feedForwardFinalResult = m_outputLayer.FeedForward(feedForwardResults);
	return feedForwardFinalResult;
#endif

	return 0.0f;
}

const std::vector<float> NeuralNetwork::Recurrent(const std::vector<float> inputs, size_t iterations /*= 1*/) const
{
	std::vector<float> recurrent{};

	for (size_t iH = 0; iH < iterations; ++iH)
		recurrent = m_hiddenLayer.Recurrent(inputs);

	const std::vector<float>& finalRecurrentOutput = m_outputLayer.Recurrent(recurrent);
	return finalRecurrentOutput;
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
	std::cout << "HIDDEN LAYER" << std::endl;
	std::cout << "==============" << std::endl;

	m_hiddenLayer.PrintLayer();

	std::cout << "==============" << std::endl;
	std::cout << "END OF HIDDEN LAYER" << std::endl;
	std::cout << "==============" << std::endl;

	std::cout << "==============" << std::endl;
	std::cout << "OUTPUT LAYER" << std::endl;
	std::cout << "==============" << std::endl;

	m_outputLayer.PrintLayer();

	std::cout << "==============" << std::endl;
	std::cout << "END OF OUTPUT LAYER" << std::endl;
	std::cout << "==============" << std::endl;
}