#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{}

NeuralNetwork::NeuralNetwork(const size_t numberOfInputNeurons, const size_t numberOfHiddenNeurons, const size_t numberOfHiddenLayers, const size_t numberOfOutputNeurons) :
	m_inputLayer{ numberOfInputNeurons },
	m_hiddenLayers{},
	m_outputLayer{ numberOfOutputNeurons, {} },
	m_finalResult{ 0.0f },
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
	
	// 
	// HIDDEN LAYER
	//

	// Pass sum of weights over to the hidden layer
	const std::vector<Neuron>& inputNeurons{ m_inputLayer.GetNeurons() };
	std::vector<float> hiddenWeightsIn(inputNeurons.size(), 0.0f);
	for (size_t iN = 0; iN < inputNeurons.size(); ++iN)
	{
		const Neuron& neuron = inputNeurons[iN];
		float sumOfWeights = neuron.GetSum();
		hiddenWeightsIn[iN] = sumOfWeights;
	}

	std::vector<float> previousHiddenLayerWeights{};
	for (size_t iHL = 0; iHL < m_numberOfHiddenLayers; ++iHL)
	{
		if (iHL == 0)
		{
			HiddenLayer hiddenLayer{ m_numberOfHiddenNeurons, hiddenWeightsIn };
			hiddenLayer.InitLayer();
			m_hiddenLayers.push_back(hiddenLayer);
		}
		else
		{
			const std::vector<Neuron>& previousHiddenLayerNeurons = m_hiddenLayers[iHL - 1].GetNeurons();
			for (const Neuron& neuron : previousHiddenLayerNeurons)
			{
				const float sum = neuron.GetSum();
				previousHiddenLayerWeights.push_back(sum);
			}

			HiddenLayer hiddenLayer{ m_numberOfHiddenNeurons, previousHiddenLayerWeights };
			hiddenLayer.InitLayer();
			m_hiddenLayers.push_back(hiddenLayer);

			// Only clear this vector if we have another hidden layer to move to
			if(iHL != m_numberOfHiddenLayers - 1)
				previousHiddenLayerWeights.clear();
		}
	}

	// OUTPUT LAYER
	OutputLayer ol{ 5, previousHiddenLayerWeights };
	m_outputLayer = ol;
	m_outputLayer.InitLayer();
	m_finalResult = m_outputLayer.GetSumOfNeuronWeights();
}

void NeuralNetwork::Process()
{

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


	std::cout << "Final Result = " << m_finalResult << std::endl;
}