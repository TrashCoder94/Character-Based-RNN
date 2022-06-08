#include "RNN.h"
#include "Utilities.h"

RNN::RNN() :
	m_inputSize{ 0 },
	m_hiddenSize{ 0 },
	m_outputSize{ 0 },
	m_learningRate{ 0.0f },
	m_h{},
	m_Wxh{},
	m_Whh{},
	m_Why{},
	m_bh{},
	m_by{},
	m_adaWxh{},
	m_adaWhh{},
	m_adaWhy{},
	m_adabh{},
	m_adaby{}
{}

RNN::~RNN()
{}

void RNN::Init(const size_t inputSize, const size_t outputSize, const size_t hiddenSize, const float learningRate)
{
	m_inputSize = inputSize;
	m_hiddenSize = hiddenSize;
	m_outputSize = outputSize;
	m_learningRate = learningRate;

	Utilities::Zeros(m_hiddenSize, 1, m_h);
	Utilities::InitialiseRandomVector(m_hiddenSize, m_inputSize, 0.01f, m_Wxh);
	Utilities::InitialiseRandomVector(m_hiddenSize, m_hiddenSize, 0.01f, m_Whh);
	Utilities::InitialiseRandomVector(m_outputSize, m_hiddenSize, 0.01f, m_Why);

	Utilities::Zeros(m_hiddenSize, 1, m_bh);
	Utilities::Zeros(m_outputSize, 1, m_by);

	Utilities::Zeros(m_hiddenSize, m_inputSize, m_adaWxh);
	Utilities::Zeros(m_hiddenSize, m_hiddenSize, m_adaWhh);
	Utilities::Zeros(m_outputSize, m_hiddenSize, m_adaWhy);
	Utilities::Zeros(m_hiddenSize, 1, m_adabh);
	Utilities::Zeros(m_outputSize, 1, m_adaby);
}

float RNN::Train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets)
{
	float loss{ 0.0f };

	/*
	#=====initialize=====
        xhat = {}#holds 1-of-k representations of x
        yhat = {}#holds 1-of-k representations of predicted y (unnormalized log probs)
        p = {}#the normalized probabilities of each output through time
        h = {}#holds state vectors through time
        h[-1] = np.copy(self.h)#we will need to access the previous state to calculate the current state
	*/
	std::vector<std::vector<float>> xHat{};	Utilities::Zeros(inputs.size(), inputs[0].size(), xHat);
	std::vector<std::vector<float>> yHat{};	Utilities::Zeros(inputs.size(), inputs[0].size(), yHat);
	std::vector<std::vector<float>> p{};	Utilities::Zeros(inputs.size(), inputs[0].size(), p);
	std::vector<std::vector<float>> h{};	Utilities::Zeros(inputs.size(), inputs[0].size(), h);

	//h[h.size() - 1] = ;
	for (size_t i = 0; i < m_h.size(); ++i)
	{
		bool isLastElement{ i == (m_h.size() - 1) };
		std::vector<float> hVec{};
		std::vector<float> mhVec{ m_h[i] };
		for (size_t j = 0; j < mhVec.size(); ++j)
		{
			if (isLastElement)
			{
				hVec.push_back(mhVec[j]);
			}
			else
			{
				hVec.push_back(0.0f);
			}
		}
		h.push_back(hVec);
	}

	/*
		dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(self.h)
	*/
	std::vector<std::vector<float>> dWxh{};		Utilities::Zeros(m_hiddenSize, m_inputSize, dWxh);
	std::vector<std::vector<float>> dWhh{};		Utilities::Zeros(m_hiddenSize, m_hiddenSize, dWhh);
	std::vector<std::vector<float>> dWhy{};		Utilities::Zeros(m_outputSize, m_hiddenSize, dWhy);
	std::vector<std::vector<float>> dbh{};		Utilities::Zeros(m_hiddenSize, 1, dbh);
	std::vector<std::vector<float>> dby{};		Utilities::Zeros(m_outputSize, 1, dby);
	std::vector<std::vector<float>> dhnext{};	Utilities::Zeros(m_hiddenSize, 1, dhnext);

	/*
	#=====forward pass=====
        loss = 0
        for t in range(len(x)):
            xhat[t] = np.zeros((self.insize, 1))
            xhat[t][x[t]] = 1#xhat[t] = 1-of-k representation of x[t]

            h[t] = np.tanh(np.dot(self.W_xh, xhat[t]) + np.dot(self.W_hh, h[t-1]) + self.b_h)#find new hidden state
            yhat[t] = np.dot(self.W_hy, h[t]) + self.b_y#find unnormalized log probabilities for next chars

            p[t] = np.exp(yhat[t]) / np.sum(np.exp(yhat[t]))#find probabilities for next chars

            loss += -np.log(p[t][y[t],0])#softmax (cross-entropy loss)
	*/

	for (size_t i = 0; i < inputs.size(); ++i)
	{
		/*
		// Input State
		xs[iN] = std::vector<float>(vocabSize, 0.0f); // one hot encode
		xs[iN][inputs[iN]] = 1.0f;
		*/
		std::vector<std::vector<float>> xHatVec{ xHat[i] };
		Utilities::Zeros(m_inputSize, 1, xHatVec);
	}

	return loss;
}