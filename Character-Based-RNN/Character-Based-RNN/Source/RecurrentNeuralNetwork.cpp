#include "RecurrentNeuralNetwork.h"
#include <array>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <sstream>

#define REGISTER_ALL_CHARACTERS 0

namespace RNNHelpers
{
	static std::default_random_engine s_randomEngine{};
}

RecurrentNeuralNetwork::RecurrentNeuralNetwork(const fs::path& inputFile, const fs::path& outputFile) :
	dictionaryCharToID{},
	dictionaryIDToChar{},
	inputFilepath{ inputFile },
	outputFilepath{ outputFile },
	inputString{""},
	outputString{""},
	hiddenLayerSize{512},
	sequenceLength{100},
	learningRate{0.5f},
	inputToHiddenWeight{0.0f},
	hiddenToHiddenWeight{0.0f},
	hiddenToOutputWeight{0.0f},
	hiddenBias{0.0f},
	outputBias{0.0f},
	cycles{ 1000 },
	isRunning{false}
{}

RecurrentNeuralNetwork::~RecurrentNeuralNetwork()
{}

void RecurrentNeuralNetwork::ReadFile(const fs::path file, bool shouldBeClean /* = false */)
{
	// Clear any current input strings.
	inputString.clear();

	// Open input file
	std::ifstream inputFile{ file };
	if (inputFile.is_open())
	{
		std::string line{ "" };

		// Read the input file line by line
		while (std::getline(inputFile, line))
		{
			// If we want this to be "clean" then make sure there are no new lines
			// This will create one long sequence of strings
			const std::string stringToAppend = shouldBeClean ? line + " " : line + "\n";
			inputString.append(stringToAppend);
		}
		inputFile.close();
	}
}

void RecurrentNeuralNetwork::CreateDictionary()
{
	int characterIndex = 0;

	for (char character : inputString)
	{
#if !REGISTER_ALL_CHARACTERS
		if (dictionaryCharToID.find(character) != dictionaryCharToID.end())
			continue;
#endif
		dictionaryCharToID.emplace(character, characterIndex);
		dictionaryIDToChar.emplace(characterIndex, character);
		++characterIndex;
	}

	std::cout << "Vocab size " << dictionaryCharToID.size() << std::endl;

	static std::uniform_real_distribution<float> randInputToHiddenDis(-(float)hiddenLayerSize, -(float)dictionaryCharToID.size());
	static std::uniform_real_distribution<float> randHiddenToHiddenDis((float)hiddenLayerSize, (float)hiddenLayerSize);
	static std::uniform_real_distribution<float> randHiddenToOutputDis((float)dictionaryCharToID.size(), (float)hiddenLayerSize);
	
	inputToHiddenWeight	 = randHiddenToHiddenDis(RNNHelpers::s_randomEngine) * -0.01f;
	hiddenToHiddenWeight = randHiddenToHiddenDis(RNNHelpers::s_randomEngine) * 0.01f;
	hiddenToOutputWeight = randHiddenToOutputDis(RNNHelpers::s_randomEngine) * 0.01f;

	hiddenBias = 0.0f;
	outputBias = 0.0f;
}

void RecurrentNeuralNetwork::Loss(std::vector<int>& inputs, std::vector<int>& target, float& outLoss, 
	std::vector<float>& outDeltaInputToHiddenWeights, std::vector<float>& outDeltaHiddenToHiddenWeights,
	std::vector<float>& outDeltaHiddenToOutputWeights, float& outDeltaHiddenBias, float& outDeltaOutputBias,
	float& outHiddenState)
{
	float loss{ 0.0f };

	std::vector<float> xs(inputs.size(), 0.0f);
	std::vector<float> hs(inputs.size(), 0.0f);
	std::vector<float> ys(inputs.size(), 0.0f);
	std::vector<float> ps(inputs.size(), 0.0f);

	for (size_t iC = 0; iC < inputs.size(); ++iC)
	{
		xs[iC] = 1.0f;

		hs[iC] = { std::tanh(Dot(inputToHiddenWeight, xs[iC]) + Dot(hiddenToHiddenWeight, hs[(iC == 0 ? inputs.size() - 1 : iC - 1)]) + hiddenBias) };
		ys[iC] = { Dot(hiddenToOutputWeight, hs[iC]) + outputBias };

		float ysTotal{ 0.0f };
		for (float& val : ys)
			ysTotal += val;

		ps[iC] = { std::exp(ys[iC]) / std::exp(ysTotal) };
		int targetVal = target[iC];
		loss += (-std::log(ps[iC] * (float)targetVal));
		//loss += -(std::log(ps[iC][target[iC]]));
	}

	std::vector<float> deltaInputToHiddenWeight(inputs.size(), inputToHiddenWeight);
	std::vector<float> deltaHiddenToHiddenWeight(inputs.size(), hiddenToHiddenWeight);
	std::vector<float> deltaHiddenToOutputWeight(inputs.size(), hiddenToOutputWeight);
	float deltaHiddenBias{ hiddenBias };
	float deltaOutputBias{ outputBias };
	std::vector<float> deltaHNext(inputs.size(), hs[0]);

	for (size_t iC = inputs.size() - 1; iC > 0; --iC)
	{
		const float deltaY{ ps[iC] - 1.0f };
		deltaHiddenToOutputWeight[iC] += Dot(deltaY, hs[iC]);
		deltaOutputBias += deltaY;

		const float deltaH{ Dot(deltaHiddenToOutputWeight[iC], deltaY) + deltaHNext[iC] };
		const float deltaHRaw{ (1.0f - hs[iC] * hs[iC]) * deltaH };

		deltaHiddenBias += deltaHRaw;
		deltaInputToHiddenWeight[iC] += Dot(deltaHRaw, xs[iC]);
		deltaHiddenToHiddenWeight[iC] += Dot(deltaHRaw, hs[iC - 1]);
		deltaHNext[iC] = Dot(hiddenToHiddenWeight, deltaHRaw);
	}

	for (size_t dP = 0; dP < deltaInputToHiddenWeight.size(); ++dP)
	{
		deltaInputToHiddenWeight[dP] = std::clamp(deltaInputToHiddenWeight[dP], -5.0f, 5.0f);
		deltaHiddenToHiddenWeight[dP] = std::clamp(deltaHiddenToHiddenWeight[dP], -5.0f, 5.0f);
		deltaHiddenToOutputWeight[dP] = std::clamp(deltaHiddenToOutputWeight[dP], -5.0f, 5.0f);
		deltaHiddenBias = std::clamp(deltaHiddenBias, -5.0f, 5.0f);
		deltaOutputBias = std::clamp(deltaOutputBias, -5.0f, 5.0f);
	}

	outLoss = loss;
	outDeltaInputToHiddenWeights = deltaInputToHiddenWeight;
	outDeltaHiddenToHiddenWeights = deltaHiddenToHiddenWeight;
	outDeltaHiddenToOutputWeights = deltaHiddenToOutputWeight;
	outDeltaHiddenBias = deltaHiddenBias;
	outDeltaOutputBias = deltaOutputBias;
	outHiddenState = hs[inputs.size() - 1];
}

void RecurrentNeuralNetwork::Sample(float& h, const size_t& seed_ix, const size_t numberOfSamples, std::vector<int>& outSamples)
{
	std::vector<float> x(dictionaryCharToID.size(), 0.0f);
	x[seed_ix] = 1.0f;

	for (size_t iS = 0; iS < numberOfSamples; ++iS)
	{
		h = std::tanh(Dot(inputToHiddenWeight, x[0]) + Dot(hiddenToHiddenWeight, h) + hiddenBias);
		const float y = Dot(hiddenToOutputWeight, h) + outputBias;

		float totalX{0.0f};
		for (float value : x)
			totalX += value;

		const float p = std::exp(y) / std::exp(totalX);
		static std::uniform_real_distribution<float> randIXDis(p, (float)dictionaryCharToID.size());
		const float iX = randIXDis(RNNHelpers::s_randomEngine);
		x = std::vector<float>(dictionaryCharToID.size(), 0.0f);
		x[(size_t)iX] = 1.0f;
		outSamples.push_back((int)iX);
	}
}

void RecurrentNeuralNetwork::Process()
{
	int n = 0;
	int p = 0;
	std::vector<float> mInputToHiddenWeight(dictionaryCharToID.size(), inputToHiddenWeight);
	std::vector<float> mHiddenToHiddenWeight(dictionaryCharToID.size(), hiddenToHiddenWeight);
	std::vector<float> mHiddenToOutputWeight(dictionaryCharToID.size(), hiddenToOutputWeight);
	float mHiddenBias{ hiddenBias };
	float mOutputBias{ outputBias };
	float smoothLoss = -(std::log(1.0f / dictionaryCharToID.size()) * sequenceLength);
	isRunning = true;

	//std::vector<float> hPrev(hiddenLayerSize, 0.0f);
	float hPrev{ 0.0f };

	std::ofstream outputFile{ outputFilepath };
	if (outputFile.is_open())
	{
		outputFile.clear();

		while (isRunning)
		{
			if (p + sequenceLength + 1 >= inputString.size() || n == 0)
				p = 0;

			std::vector<int> inputs;
			for (size_t iC = p; iC < p + sequenceLength; ++iC)
			{
				const int ID = dictionaryCharToID.at(inputString[iC]);
				inputs.push_back(ID);
			}

			std::vector<int> targets;
			for (size_t iC = p + 1; iC < p + sequenceLength + 1; ++iC)
			{
				const int ID = dictionaryCharToID.at(inputString[iC]);
				targets.push_back(ID);
			}

			if ((n % 100) == 0)
			{
				std::vector<int> samples{};
				Sample(hPrev, inputs[0], 200, samples);

				std::string sampleString{};
				for (size_t iS = 0; iS < samples.size(); ++iS)
				{
					int sampleIndex{ samples[iS] };
					const char character = dictionaryIDToChar.at(sampleIndex);
					std::cout << "----\n" << character << "\n----" << std::endl;
					sampleString += character;
				}

				outputFile << sampleString;
			}

			float loss{ 0.0f };
			std::vector<float> deltaInputToHidden{};
			std::vector<float> deltaHiddenToHidden{};
			std::vector<float> deltaHiddenToOutput{};
			float deltaHiddenBias{ 0.0f };
			float deltaOutputBias{ 0.0f };

			Loss(inputs, targets, loss, deltaInputToHidden, deltaHiddenToHidden, deltaHiddenToOutput
				, deltaHiddenBias, deltaOutputBias, hPrev);
			smoothLoss = smoothLoss * 0.999f + loss * 0.001f;

			if (n % 100 == 0)
				std::cout << "Iteration: " << n << " Loss: " << smoothLoss << std::endl;

			std::vector<float> firstLayerData{ inputToHiddenWeight, hiddenToHiddenWeight, hiddenToOutputWeight, hiddenBias, outputBias };
			std::vector<float> secondLayerData{};
			std::vector<float> thirdLayerData{};

			secondLayerData.insert(std::end(secondLayerData), std::begin(deltaInputToHidden), std::end(deltaInputToHidden));
			secondLayerData.insert(std::end(secondLayerData), std::begin(deltaHiddenToHidden), std::end(deltaHiddenToHidden));
			secondLayerData.insert(std::end(secondLayerData), std::begin(deltaHiddenToOutput), std::end(deltaHiddenToOutput));
			secondLayerData.push_back(deltaHiddenBias);
			secondLayerData.push_back(deltaOutputBias);

			thirdLayerData.insert(std::end(thirdLayerData), std::begin(mInputToHiddenWeight), std::end(mInputToHiddenWeight));
			thirdLayerData.insert(std::end(thirdLayerData), std::begin(mHiddenToHiddenWeight), std::end(mHiddenToHiddenWeight));
			thirdLayerData.insert(std::end(thirdLayerData), std::begin(mHiddenToOutputWeight), std::end(mHiddenToOutputWeight));
			thirdLayerData.push_back(mHiddenBias);
			thirdLayerData.push_back(mOutputBias);

			for (float& param : firstLayerData)
			{
				for (float& dParam : secondLayerData)
				{
					for (float& mem : thirdLayerData)
					{
						mem += (dParam * dParam);
						param += ((-learningRate * dParam) / std::sqrt(mem + (float)1e-8));
					}
				}
			}

			p += sequenceLength;
			++n;

			if (n % cycles == 0)
			{
				std::cout << "Finished " << cycles << " cycles!" << std::endl;
				isRunning = false;
			}
		}
	
		outputFile.close();
	}

	/*
	while True:			  
	  # perform parameter update with Adagrad
	  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
	                                [dWxh, dWhh, dWhy, dbh, dby], 
	                                [mWxh, mWhh, mWhy, mbh, mby]):
	    mem += dparam * dparam
	    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
	*/
}

float RecurrentNeuralNetwork::Dot(const float& a, const float& b)
{
	float dot{0.0f};
	dot = dot + (a * b);
	return dot;
}