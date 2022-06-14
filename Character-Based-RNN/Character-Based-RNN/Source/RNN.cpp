#include "RNN.h"
#include "Utilities.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <thread>

// 1 means delta[i] = data;
// 0 means delta.push_back(data);
#define DEBUG_SET_DELTA_RNNDATA 1

RNN::RNN() :
	m_textData{},
	m_textDataSize{0},
	m_vocabSize{0},
	m_charToID{},
	m_IDToChar{},
	m_numberOfIterations{0},
	m_sequenceLength{0},
	m_batchSize{0},
	m_hiddenSize{0},
	m_learningRate{0.0f},
	m_Wxh{},
	m_Whh{},
	m_Why{},
	m_bh{},
	m_by{},
	m_hPrev{},
	m_adaWxh{},
	m_adaWhh{},
	m_adaWhy{},
	m_adaBh{},
	m_adaBy{}
{}

RNN::~RNN()
{}

void RNN::LoadTextData(const fs::path dataFilepath)
{
	/*
	# load text data
	txt_data = "abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz " # input data
	# txt_data = open('input.txt', 'r').read() # test external files
	*/
	m_textData = Utilities::ReadFile(dataFilepath);
	m_textDataSize = m_textData.size();
}

void RNN::OneHotEncode()
{
	/*
	# one hot encode
	char_to_int = dict((c, i) for i, c in enumerate(chars)) # "enumerate" retruns index and value. Convert it to dictionary
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	print(char_to_int)
	print("----------------------------------------------------")
	print(int_to_char)
	print("----------------------------------------------------")
	# integer encode input data
	integer_encoded = [char_to_int[i] for i in txt_data] # "integer_encoded" is a list which has a sequence converted from an original data to integers.
	print(integer_encoded)
	print("----------------------------------------------------")
	print("data length : ", len(integer_encoded))
	*/
	int charID{ 0 };
	for (char character : m_textData)
	{
		if (m_charToID.find(character) != m_charToID.end())
			continue;

		m_charToID.emplace(character, charID);
		m_IDToChar.emplace(charID, character);
		++charID;
	}
	m_vocabSize = m_charToID.size();

	for (auto mapVal : m_charToID)
	{
		std::cout << mapVal.first << " : " << mapVal.second << " ";
	}
	std::cout << std::endl;

	for (auto mapVal : m_IDToChar)
	{
		std::cout << mapVal.first << " : " << mapVal.second << " ";
	}
	std::cout << std::endl;

	/*
	print("unique characters : ", num_chars) # You can see the number of unique characters in your input data.
	print("txt_data_size : ", txt_data_size)
	*/
	std::cout << "Unique Characters: " << m_vocabSize << std::endl;
	std::cout << "Text Data Size: " << m_textDataSize << std::endl << std::endl;
}

void RNN::Init(const fs::path dataFilepath, const size_t iterations, const size_t sequenceLength, const size_t hiddenSize, const float learningRate)
{
	std::cout << "RNN Initialisation" << std::endl;
	std::cout << "==================" << std::endl;

	LoadTextData(dataFilepath);
	OneHotEncode();

	/*
	iteration = 5000
	sequence_length = 10
	batch_size = round((txt_data_size /sequence_length)+0.5) # = math.ceil
	hidden_size = 100  # size of hidden layer of neurons.  
	learning_rate = 1e-1
	*/
	m_numberOfIterations = iterations;
	m_sequenceLength = sequenceLength;
	m_batchSize = (size_t)std::round((m_textDataSize / m_sequenceLength) + 0.5f);
	m_hiddenSize = hiddenSize;
	m_learningRate = learningRate;

	/*
	W_xh = np.random.randn(hidden_size, num_chars)*0.01     # weight input -> hidden. 
	W_hh = np.random.randn(hidden_size, hidden_size)*0.01   # weight hidden -> hidden
	W_hy = np.random.randn(num_chars, hidden_size)*0.01     # weight hidden -> output
	b_h = np.zeros((hidden_size, 1)) # hidden bias
	b_y = np.zeros((num_chars, 1)) # output bias
	h_prev = np.zeros((hidden_size,1)) # h_(t-1)
	*/
	Utilities::InitialiseRandomVectorOfFloatVectors(m_hiddenSize, m_vocabSize, 0.01f, m_Wxh);
	Utilities::InitialiseRandomVectorOfFloatVectors(m_hiddenSize, m_hiddenSize, 0.01f, m_Whh);
	Utilities::InitialiseRandomVectorOfFloatVectors(m_vocabSize, m_hiddenSize, 0.01f, m_Why);
	Utilities::Zeros(m_hiddenSize, 1, m_bh);
	Utilities::Zeros(m_vocabSize, 1, m_by);
	Utilities::Zeros(m_hiddenSize, 1, m_hPrev);
}

float RNN::ForwardPropagation(std::vector<int>& inputs, std::vector<int>& targets, rnnData& hPrev, rnnData& outPs, rnnData& outHs, rnnData& outXs)
{
	/*
	# Since the RNN receives the sequence, the weights are not updated during one sequence.
	xs, hs, ys, ps = {}, {}, {}, {} # dictionary
	hs[-1] = np.copy(h_prev) # Copy previous hidden state vector to -1 key value.
	loss = 0 # loss initialization
	*/
	Utilities::Zeros(inputs.size(), m_vocabSize, outXs);
	Utilities::Zeros(inputs.size(), m_vocabSize, outHs);
	rnnData ys{};	
	Utilities::Zeros(inputs.size(), m_vocabSize, ys);
	Utilities::Zeros(inputs.size(), m_vocabSize, outPs);
	outHs[outHs.size() - 1] = hPrev[0];
	float loss{ 0.0f };

	for (size_t i = 0; i < inputs.size(); ++i)
	{
		/*
		xs[t] = np.zeros((num_chars,1)) 
        xs[t][inputs[t]] = 1
		*/
		outXs[i] = std::vector<float>(m_vocabSize, 0.0f);
		outXs[i][inputs[i]] = 1.0f;

		// hs[t] = np.tanh(np.dot(W_xh, xs[t]) + np.dot(W_hh, hs[t-1]) + b_h)
		if (i == 0)
		{
			// Always assume hidden state will start at 0
			outHs[i] = std::vector<float>(m_vocabSize, 0.0f);
		}
		else
		{
			outHs[i] = Utilities::CalculateHiddenState(m_Wxh, outXs[i], m_Whh, outHs[i - 1], m_bh);
		}

		// ys[t] = np.dot(W_hy, hs[t]) + b_y
		ys[i] = Utilities::CalculateUnormalizedProbabilitiesForNextChars(m_Why, outHs[i], m_by);
		
		// ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
		outPs[i] = Utilities::CalculateProbabilitiesForNextChars(ys[i]);

		// loss += -np.log(ps[t][targets[t],0])
		const float probability{ outPs[i][targets[i]] };
		loss += -std::log(probability);
	}

	return loss;
}

void RNN::BackwardPropagation(std::vector<int>& inputs, std::vector<int>& targets, rnnData& ps, rnnData& xs, rnnData& hs, rnnData& outDWxh, rnnData& outDWhh, rnnData& outDWhy, rnnData& outDbh, rnnData& outDby)
{
	/*
	dWxh, dWhh, dWhy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy) # make all zero matrices.
    dbh, dby = np.zeros_like(b_h), np.zeros_like(b_y)
    dhnext = np.zeros_like(hs[0]) # (hidden_size,1) 
	*/
	Utilities::Zeros(m_Wxh.size(), m_Wxh[0].size(), outDWxh);
	Utilities::Zeros(m_Whh.size(), m_Whh[0].size(), outDWhh);
	Utilities::Zeros(m_Why.size(), m_Why[0].size(), outDWhy);
	Utilities::Zeros(m_bh.size(), m_bh[0].size(), outDbh);
	Utilities::Zeros(m_by.size(), m_by[0].size(), outDby);
	std::vector<float> dhNext{};
	Utilities::Zeros(hs[0].size(), dhNext);

	const rnnData hsTransposed{ Utilities::Transpose(hs) };
	const rnnData whyTransposed{ Utilities::Transpose(m_Why) };
	const rnnData xsTransposed{ Utilities::Transpose(xs) };
	const rnnData whhTransposed{ Utilities::Transpose(m_Whh) };

	for (size_t i = inputs.size() - 1; i > 0; --i)
	{
		/*
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
		dWhy += np.dot(dy, hs[t].T)
		dby += dy
		dh = np.dot(Why.T, dy) + dhnext # backprop into h
		dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
		dbh += dhraw
		dWxh += np.dot(dhraw, xs[t].T)
		dWhh += np.dot(dhraw, hs[t-1].T)
		dhnext = np.dot(Whh.T, dhraw)
		*/

		//  dy = np.copy(ps[t]) # shape (num_chars,1).  "dy" means "dloss/dy"
		std::vector<float> dy = ps[i];
		
		// dy[targets[t]] -= 1 # backprop into y. After taking the soft max in the input vector, subtract 1 from the value of the element corresponding to the correct label.
		dy[targets[i]] -= 1.0f;

		// dWhy += np.dot(dy, hs[t].T)
#if DEBUG_SET_DELTA_RNNDATA
		auto a{ Utilities::Dot(dy, hsTransposed[i]) };
		for (size_t j = 0; j < a.size(); ++j)
		{
			outDWhy[i][j] = a[j];
		}
#else
		outDWhy.push_back(Utilities::Dot(dy, hsTransposed[i]));
#endif

		// dby += dy
#if DEBUG_SET_DELTA_RNNDATA
		/*for (size_t j = 0; j < outDby.size(); ++j)
		{
			outDby[j][0] = dy[j];
		}*/
		outDby[i][0] = dy[i];
#else
		outDby.push_back(dy);
#endif

		// dh = np.dot(W_hy.T, dy) + dhnext # backprop into h. 
		std::vector<float> dh{ Utilities::BackPropIntoDeltaH(whyTransposed, dy, dhNext) };

		// dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
		std::vector<float> dhRaw{ Utilities::BackPropThroughTanh(hs[i], dh) };

		// dbh += dhraw
#if DEBUG_SET_DELTA_RNNDATA
		outDbh[i][0] = dhRaw[i];
		//outDbh[i] = dhRaw;
#else
		outDbh.push_back(dhRaw);
#endif

		// dWxh += np.dot(dhraw, xs[t].T)
#if DEBUG_SET_DELTA_RNNDATA
		auto b{ Utilities::Dot(dhRaw, xsTransposed[i]) };
		for (size_t j = 0; j < b.size(); ++j)
		{
			outDWxh[i][j] = b[j];
		}
		//outDWxh[i] = Utilities::Dot(dhRaw, xsTransposed[i]);
#else
		outDWxh.push_back(Utilities::Dot(dhRaw, xsTransposed[i]));
#endif

		// dWhh += np.dot(dhraw, hs[t-1].T)
#if DEBUG_SET_DELTA_RNNDATA
		auto c{ Utilities::Dot(dhRaw, hsTransposed[i - 1]) };
		for (size_t j = 0; j < c.size(); ++j)
		{
			outDWhh[i][j] = c[j];
		}
		//outDWhh[i] = Utilities::Dot(dhRaw, hsTransposed[i - 1]);
#else
		outDWhh.push_back(Utilities::Dot(dhRaw, hsTransposed[i - 1]));
#endif

		// dhnext = np.dot(W_hh.T, dhraw)
		dhNext = Utilities::Dot(whhTransposed, dhRaw);
	}

	const float clipVal{ 1.0f };
	Utilities::Clip(outDWxh, -clipVal, clipVal);
	Utilities::Clip(outDWhh, -clipVal, clipVal);
	Utilities::Clip(outDWhy, -clipVal, clipVal);
	Utilities::Clip(outDbh, -clipVal, clipVal);
	Utilities::Clip(outDby, -clipVal, clipVal);
}

using namespace std::chrono_literals;
#define JASON_TIME_TEST 1

void RNN::Training()
{
	// std::cout << "\r" << std::string(percent5, '|') << percent5*5 << "%";
	auto func = [&](std::atomic<bool>& bRun, std::atomic<float>& curItr, const float& maxItr)
	{
		while (bRun.load())
		{
			float currentIteration{ curItr.load() };
			float normalizedProgress{ (currentIteration / maxItr) };
			float currentProgress{ normalizedProgress * 100.0f };
			std::cerr << "Progress: " << currentProgress << "% \r";
			std::this_thread::sleep_for(1000ms);
		}
	};
	std::atomic<float> currentIteration{ 0.0f };
	float maxNumberOfIterations{ (float)m_numberOfIterations };
	std::atomic<bool> run(true);
	std::thread asyncThread(func, std::ref(run), std::ref(currentIteration), std::ref(maxNumberOfIterations));

	std::cout << std::endl;
	std::cout << "RNN Training" << std::endl;
	std::cout << "============" << std::endl;
	std::cout << "This may take some time..." << std::endl << std::endl;

	/*
	data_pointer = 0

	# memory variables for Adagrad
	mWxh, mWhh, mWhy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
	mbh, mby = np.zeros_like(b_h), np.zeros_like(b_y) 
	*/
	size_t dataPointer{ 0 };
	Utilities::Zeros(m_Wxh.size(), m_Wxh[0].size(), m_adaWxh);
	Utilities::Zeros(m_Whh.size(), m_Whh[0].size(), m_adaWhh);
	Utilities::Zeros(m_Why.size(), m_Why[0].size(), m_adaWhy);
	Utilities::Zeros(m_bh.size(), m_bh[0].size(), m_adaBh);
	Utilities::Zeros(m_by.size(), m_by[0].size(), m_adaBy);
	
	//float smoothLoss{ -np.log(1.0 / vocab_size) * seq_length }
	float smoothLoss{ -std::log(1.0f / (float)m_vocabSize) * (float)m_sequenceLength };
	auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < m_numberOfIterations; ++i)
	{
		Utilities::Zeros(m_hiddenSize, 1, m_hPrev);
		dataPointer = 0;

		for (size_t j = 0; j < m_batchSize; ++j)
		{
			std::vector<int> inputs{};
			for (size_t iN = dataPointer; iN < dataPointer + m_sequenceLength; ++iN)
			{
				if (iN > m_textDataSize)
					continue;
				char character{ m_textData[iN] };
				if (m_charToID.find(character) == m_charToID.end())
					continue;
				int ID{ m_charToID.at(character) };
				inputs.push_back(ID);
			}
			std::vector<int> targets{};
			for (size_t iN = dataPointer + 1; iN < dataPointer + m_sequenceLength + 1; ++iN)
			{
				if (iN > m_textDataSize)
					continue;
				char character{ m_textData[iN] };
				if (m_charToID.find(character) == m_charToID.end())
					continue;
				int ID{ m_charToID.at(character) };
				targets.push_back(ID);
			}

			/*
			if (data_pointer+sequence_length+1 >= len(txt_data) and b == batch_size-1): # processing of the last part of the input data.
				targets.append(char_to_int[" "])   # When the data doesn't fit, add space(" ") to the back.
			*/
			if (dataPointer + m_sequenceLength + 1 >= m_textDataSize && j == m_batchSize - 1)
			{
				char character{ ' ' };
				int ID{ m_charToID.at(character) };
				targets.push_back(ID);
			}

			/*
			# forward
			loss, ps, hs, xs = forwardprop(inputs, targets, h_prev)
			*/
			rnnData ps{};
			rnnData hs{};
			rnnData xs{};
			float loss = ForwardPropagation(inputs, targets, m_hPrev, ps, hs, xs);
			smoothLoss = smoothLoss * 0.999f + loss * 0.001f;

			/*
				# backward
				dWxh, dWhh, dWhy, dbh, dby = backprop(ps, inputs, hs, xs)
			*/
			rnnData dWxh{};	rnnData dWhh{};	rnnData dWhy{};
			rnnData dbh{};	rnnData dby{};
			BackwardPropagation(inputs, targets, ps, xs, hs, dWxh, dWhh, dWhy, dbh, dby);

			/*
			# perform parameter update with Adagrad
			for param, dparam, mem in zip([W_xh, W_hh, W_hy, b_h, b_y], e.g. m_Wxh
										[dWxh, dWhh, dWhy, dbh, dby],	e.g. dWxh
										[mWxh, mWhh, mWhy, mbh, mby]):	e.g. m_adaWxh
				mem += dparam * dparam # elementwise
				param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update 
			*/
			Utilities::AdagradUpdate(
				m_Wxh,	dWxh,	m_adaWxh, 
				m_Whh,	dWhh,	m_adaWhh,
				m_Why,	dWhy,	m_adaWhy,
				m_bh,	dbh,	m_adaBh,
				m_by,	dby,	m_adaBy, 
				m_learningRate);
			
			dataPointer += m_sequenceLength;
		}

		/*
		if i % 100 == 0:
		print('iter %d, loss: %f' % (i, loss)) # print progress
		*/
		currentIteration.store((float)i);

		if (i > 0 && i % 100 == 0)
		{
			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<float> elapsedSeconds = end - start;
			const float durationInSeconds{ elapsedSeconds.count() };
			std::cout << "Iter: " << i << " Loss: " << smoothLoss << " Duration: " << durationInSeconds << "s" << std::endl;
			start = std::chrono::steady_clock::now();
		}
	}

	run.store(false);
	asyncThread.join();
}

void RNN::Generate(char character, size_t lengthOfWordToGenerate, fs::path outputFilepath)
{
	/*
	x = np.zeros((num_chars, 1)) 
    x[char_to_int[test_char]] = 1
    ixes = []
    h = np.zeros((hidden_size,1))
	*/
	rnnData x{};
	rnnData h{};
	Utilities::Zeros(m_vocabSize, 1, x);
	Utilities::Zeros(m_hiddenSize, 1, h);
	int ID{m_charToID.at(character)};
	x[ID][0] = 1.0f;
	std::vector<int> IDs{};

	for (size_t i = 0; i < lengthOfWordToGenerate; ++i)
	{
		// h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h) 
		h = Utilities::GenerateTanh(m_Wxh, x, m_Whh, h, m_bh);

		// y = np.dot(W_hy, h) + b_y
		rnnData y{ Utilities::GenerateUnormalizedProbabilities(m_Why, h, m_bh) };

		//   p = np.exp(y) / np.sum(np.exp(y)) 
		std::vector<float> pRavel{};
		rnnData p{ Utilities::GenerateProbabilites(y, pRavel) };

		//  ix = np.random.choice(range(num_chars), p=p.ravel()) # ravel -> rank0
		int randomID{ Utilities::GenerateRandomIndex(pRavel, m_vocabSize) };

		/*
		 x = np.zeros((num_chars, 1)) # init
        x[ix] = 1 
        ixes.append(ix) # list
		*/
		Utilities::Zeros(m_vocabSize, 1, x);
		x[randomID][0] = 1.0f;
		IDs.push_back(randomID);
	}

	/*
	 txt = ''.join(int_to_char[i] for i in ixes)
    print ('----\n %s \n----' % (txt, ))
	*/
	std::ofstream outputFile{ outputFilepath };
	if (outputFile.is_open())
	{
		std::string generatedString{};
		for (int generatedID : IDs)
		{
			char character{ m_IDToChar.at(generatedID) };
			generatedString += character;
		}

		outputFile << generatedString;
		outputFile.close();
	}
}