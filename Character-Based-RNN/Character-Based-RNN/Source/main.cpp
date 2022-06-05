// Super messy conversion of https://gist.github.com/karpathy/d4dee566867f8291f086 from python to C++!
#define DEBUG_SAMPLE_TEXT 0

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <conio.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <thread>
#include <tuple>
#include <Windows.h>

//////////////////////////////////////////////////////////////////////////

namespace fs = std::filesystem;

//////////////////////////////////////////////////////////////////////////

std::string inputFileText{ "" };
size_t dataSize{ 0 };
std::map<char, int> charToID{};
std::map<char, int> IDToChar{};
size_t vocabSize{ 0 };

const size_t hiddenSize{ 100 };
size_t sequenceLength{ 25 };
float learningRate{ 0.1f };
size_t numberOfIterationsToSample{ 1000 };

std::vector<std::vector<float>> weightsInputToHidden{};
std::vector<std::vector<float>> weightsHiddenToHidden{};
std::vector<std::vector<float>> weightsHiddenToOutput{};
std::vector<std::vector<float>> hiddenBiases{};
std::vector<std::vector<float>> outputBiases{};

//////////////////////////////////////////////////////////////////////////

void AsyncInput(std::atomic<bool>& run);
void Transpose(const std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& outDataTransposed);
std::string ReadFile(const fs::path filepath, const bool addNewLineAtEnd);
void Loss(std::vector<int> inputs, std::vector<int> targets, std::vector<std::vector<float>>& hPrev, float& outLoss,
	std::vector<std::vector<float>>& outDeltaWeightsInputToHidden, std::vector<std::vector<float>>& outDeltaWeightsHiddenToHidden,
	std::vector<std::vector<float>>& outDeltaWeightsHiddenToOutput, std::vector<std::vector<float>>& outDeltaHiddenBiases,
	std::vector<std::vector<float>>& outDeltaOutputBiases);
void Sample(std::vector<std::vector<float>>& h, int& seed_ix, const size_t numberOfSamples, std::vector<int>& outSampleIDs);
void Zip(std::vector<std::vector<float>>& vectorA, std::vector<std::vector<float>>& vectorB, std::vector<std::vector<float>>& vectorC,
	std::vector<std::vector<float>>& outVector);

//////////////////////////////////////////////////////////////////////////

void AsyncInput(std::atomic<bool>& run)
{
	std::string buffer{};
	std::vector<std::string> exitCommands
	{
		"Quit",	"quit",
		"Finish", "finish",
		"End", "end",
		"Stop", "stop"
	};

	while (run.load())
	{
		std::cin >> buffer;
		for (const std::string& exitCommand : exitCommands)
		{
			if (buffer == exitCommand)
			{
				std::cout << "Will exit the program once the current loop has finished processing characters..." << std::endl;
				run.store(false);
			}
		}
	}
}

void Transpose(const std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& outDataTransposed)
{
	if (data.size() == 0)
		return;

	outDataTransposed = std::vector(data[0].size(), std::vector<float>());

	for (size_t i = 0; i < data.size(); i++)
	{
		for (size_t j = 0; j < data[i].size(); j++)
		{
			outDataTransposed[j].push_back(data[i][j]);
		}
	}
}

std::string ReadFile(const fs::path filepath, const bool addNewLineAtEnd)
{
	std::string fileText{ "" };

	// Open input file
	std::ifstream inputFile{ filepath };
	if (inputFile.is_open())
	{
		std::string line{ "" };

		// Read the input file line by line
		while (std::getline(inputFile, line))
		{
			// If we want this to be "clean" then make sure there are no new lines
			// This will create one long sequence of strings
			const std::string stringToAppend = addNewLineAtEnd ? line + "\n" : line;
			fileText.append(stringToAppend);
		}
		inputFile.close();
	}

	return fileText;
}

void Loss(std::vector<int> inputs, std::vector<int> targets, std::vector<std::vector<float>>& hPrev, float& outLoss,
	std::vector<std::vector<float>>& outDeltaWeightsInputToHidden, std::vector<std::vector<float>>& outDeltaWeightsHiddenToHidden,
	std::vector<std::vector<float>>& outDeltaWeightsHiddenToOutput, std::vector<std::vector<float>>& outDeltaHiddenBiases,
	std::vector<std::vector<float>>& outDeltaOutputBiases)
{
	std::vector<std::vector<float>> xs(inputs.size(), { 0.0f });
	std::vector<std::vector<float>> hs(inputs.size(), { 0.0f });
	std::vector<std::vector<float>> ps(inputs.size(), { 0.0f });
	std::vector<std::vector<float>> ys(inputs.size(), { 0.0f });

	// hs[-1] = np.copy(hprev)
	std::vector<float>& lastHiddenStateVector{ hs[hs.size() - 1] };
	std::vector<float> hPrevCopy{};
	for (size_t i = 0; i < hPrev.size(); ++i)
	{
		std::vector<float> hPrevVec{ hPrev[i] };
		for (size_t j = 0; j < hPrevVec.size(); ++j)
		{
			const float hPrevValue{ hPrevVec[j] };
			hPrevCopy.push_back(hPrevValue);
		}
	}
	lastHiddenStateVector = hPrevCopy;

	float loss{ 0.0f };

	// Feed Forward
	for (size_t iN = 0; iN < inputs.size(); ++iN)
	{
		// Input State
		xs[iN] = std::vector<float>(vocabSize, 0.0f); // one hot encode
		xs[iN][inputs[iN]] = 1.0f;

		// Hidden State
		std::vector<float> inputToHiddenWeightDotInputValues{};
		for (size_t i = 0; i < weightsInputToHidden.size(); ++i)
		{
			std::vector<float>& weightsVector{ weightsInputToHidden[i] };
			for (size_t j = 0; j < weightsVector.size(); ++j)
			{
				float& weight = weightsVector[j];
				const float dot = weight * xs[iN][j];
				inputToHiddenWeightDotInputValues.push_back(dot);
			}
		}
		std::vector<float> hiddenToHiddenWeightDotValues{};
		for (size_t i = 0; i < weightsHiddenToHidden.size(); ++i)
		{
			std::vector<float>& weightsVector{ weightsHiddenToHidden[i] };
			for (size_t j = 0; j < weightsVector.size(); ++j)
			{
				float& weight = weightsVector[j];
				const float dot = weight * (iN == 0.0f ? 0.0f : hs[iN - 1][j]);
				hiddenToHiddenWeightDotValues.push_back(dot);
			}
		}
		std::vector<float> hiddenStateTanhValues{};
		for (size_t i = 0; i < inputToHiddenWeightDotInputValues.size(); ++i)
		{
			const float inputToHidden{ inputToHiddenWeightDotInputValues[i] };
			const float hiddenToHidden{ hiddenToHiddenWeightDotValues[i] };
			const float tanhValue{ std::tanh(inputToHidden + hiddenToHidden + hiddenBiases[iN][0]) };
			hiddenStateTanhValues.push_back(tanhValue);
		}
		hs[iN] = hiddenStateTanhValues;

		std::vector<float> hiddenToOutputDotValues{};
		for (size_t i = 0; i < weightsHiddenToOutput.size(); ++i)
		{
			std::vector<float>& weightsVector{ weightsHiddenToOutput[i] };
			for (size_t j = 0; j < weightsVector.size(); ++j)
			{
				const float weight = weightsVector[j];
				const float hiddenState = hs[iN][0];
				const float dot = weight * hiddenState;
				hiddenToOutputDotValues.push_back(dot);
			}
		}
		std::vector<float> outputValues{};
		for (size_t i = 0; i < hiddenToOutputDotValues.size(); ++i)
		{
			const float hiddenToOutputValue{ hiddenToOutputDotValues[i] };
			const float outputValue{ hiddenToOutputValue + outputBiases[iN][0] };
			outputValues.push_back(outputValue);
		}
		ys[iN] = outputValues;

		// ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
		std::vector<float> outputExpValues{};
		for (size_t i = 0; i < ys.size(); ++i)
		{
			std::vector<float>& outputVector = ys[i];
			for (size_t j = 0; j < outputVector.size(); ++j)
			{
				const float outputValue{ outputVector[j] };
				const float expValue{ std::exp(outputValue) };
				outputExpValues.push_back(expValue);
			}
		}
		float sumOfOutputExpValues{ 0.0f };
		for (size_t i = 0; i < outputExpValues.size(); ++i)
		{
			const float outputExpValue{ outputExpValues[i] };
			sumOfOutputExpValues += outputExpValue;
		}
		std::vector<float> probabilities{};
		for (size_t i = 0; i < outputExpValues.size(); ++i)
		{
			const float expValue{ outputExpValues[i] };
			const float probabilityValue{ expValue / sumOfOutputExpValues };
		}
		ps[iN] = ((probabilities.size() == 0) ? std::vector<float>{ 1.0f } : probabilities);

		// loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
		float psValue{ ps[iN][0] };
		loss += -std::log(psValue);
	}

	// # backward pass : compute gradients going backwards
	// dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	for (size_t i = 0; i < weightsInputToHidden.size(); ++i)
	{
		std::vector<float> dWxhVector{};
		std::vector<float> weightsInputToHiddenVector{ weightsInputToHidden[i] };
		for (size_t j = 0; j < weightsInputToHiddenVector.size(); ++j)
		{
			dWxhVector.push_back(0.0f);
		}
		outDeltaWeightsInputToHidden.push_back(dWxhVector);
	}
	for (size_t i = 0; i < weightsHiddenToHidden.size(); ++i)
	{
		std::vector<float> dWhhVector{};
		std::vector<float> weightsHiddenToHiddenVector{ weightsHiddenToHidden[i] };
		for (size_t j = 0; j < weightsHiddenToHiddenVector.size(); ++j)
		{
			dWhhVector.push_back(0.0f);
		}
		outDeltaWeightsHiddenToHidden.push_back(dWhhVector);
	}
	for (size_t i = 0; i < weightsHiddenToOutput.size(); ++i)
	{
		std::vector<float> dWhyVector{};
		std::vector<float> weightsHiddenToOutputVector{ weightsHiddenToOutput[i] };
		for (size_t j = 0; j < weightsHiddenToOutputVector.size(); ++j)
		{
			dWhyVector.push_back(0.0f);
		}
		outDeltaWeightsHiddenToOutput.push_back(dWhyVector);
	}

	// dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	for (size_t i = 0; i < hiddenBiases.size(); ++i)
	{
		std::vector<float> dbhVector{};
		std::vector<float> hiddenBiasesVector{ hiddenBiases[i] };
		for (size_t j = 0; j < hiddenBiasesVector.size(); ++j)
		{
			dbhVector.push_back(0.0f);
		}
		outDeltaHiddenBiases.push_back(dbhVector);
	}
	for (size_t i = 0; i < outputBiases.size(); ++i)
	{
		std::vector<float> dbyVector{};
		std::vector<float> outputBiasesVector{ outputBiases[i] };
		for (size_t j = 0; j < outputBiasesVector.size(); ++j)
		{
			dbyVector.push_back(0.0f);
		}
		outDeltaOutputBiases.push_back(dbyVector);
	}

	// dhnext = np.zeros_like(hs[0])
	const std::vector<float> hsVector{ hs[0] };
	std::vector<float> dhnext{};
	for (size_t i = 0; i < hsVector.size(); ++i)
	{
		dhnext.push_back(0.0f);
	}

	//for t in reversed(xrange(len(inputs)))

	std::vector<std::vector<float>> hsTransposed{};
	Transpose(hs, hsTransposed);
	std::vector<std::vector<float>> whyTransposed{};
	Transpose(weightsHiddenToOutput, whyTransposed);
	std::vector<std::vector<float>> xsTransposed{};
	Transpose(xs, xsTransposed);
	std::vector<std::vector<float>> whhTransposed{};
	Transpose(weightsHiddenToHidden, whhTransposed);
	for (size_t i = inputs.size() - 1; i > 0; --i)
	{
		// dy = np.copy(ps[t])
		std::vector<float> dy{};
		std::vector<float> psCopy{ ps[i] };
		for (float psVal : psCopy)
		{
			dy.push_back(psVal);
		}

		// dy[targets[t]] -= 1 # backprop into y.see http ://cs231n.github.io/neural-networks-case-study/#grad if confused here
		dy[0] -= 1.0f;

		// dWhy += np.dot(dy, hs[t].T)
		std::vector<float> deltaDotHiddenToOuputVector{};
		for (size_t j = 0; j < dy.size(); ++j)
		{
			const float dyValue{ dy[j] };
			const float hiddenStateTransposed{ hsTransposed[i][j] };
			const float dotValue{ dyValue * hiddenStateTransposed };
			deltaDotHiddenToOuputVector.push_back(dotValue);
		}
		// Might have to be index access instead since all elements are currently 0?
		// Maybe outDeltaWeightsHiddenToOutput[i] = deltaDotHiddenToOuputVector
		//outDeltaWeightsHiddenToOutput.push_back(deltaDotHiddenToOuputVector);
		outDeltaWeightsHiddenToOutput[i] = deltaDotHiddenToOuputVector;

		//dby += dy;
		std::vector<float> outputBiasesVector{};
		for (size_t j = 0; j < dy.size(); ++j)
		{
			const float outputBiasValue{ dy[j] };
			outputBiasesVector.push_back(outputBiasValue);
		}
		// Might have to be index access instead since all elements are currently 0?
		// Maybe outDeltaOutputBiases[i] = outputBiasesVector
		//outDeltaOutputBiases.push_back(outputBiasesVector);
		outDeltaOutputBiases[i] = outputBiasesVector;

		// dh = np.dot(Why.T, dy) + dhnext # backprop into h
		std::vector<float> dh{};
		for (size_t j = 0; j < whyTransposed.size(); ++j)
		{
			const float whyTransposedValue{ whyTransposed[j][i] };
			const float dyValue{ dy[0] };
			const float dotDhValue{ whyTransposedValue * dyValue + dhnext[j] };
			dh.push_back(dotDhValue);
		}

		// dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
		std::vector<float> dhraw{};
		std::vector<float> hsT{ hs[i] };
		for (size_t j = 0; j < hsT.size(); ++j)
		{
			const float hsTValue{ hsT[j] };
			const float dhValue{ dh[i] };
			const float dhRawValue{ (1.0f - hsTValue * hsTValue) * dhValue };
			dhraw.push_back(dhRawValue);
		}

		// dbh += dhraw
		// Might have to be index access instead since all elements are currently 0?
		// Maybe outDeltaOutputBiases[i] = dhraw
		//outDeltaOutputBiases.push_back(dhraw);
		outDeltaOutputBiases[i] = dhraw;

		// dWxh += np.dot(dhraw, xs[t].T)
		std::vector<float> dWxhVector{};
		for (size_t j = 0; j < dhraw.size(); ++j)
		{
			const float dhRawValue{ dhraw[j] };
			const float xsTValue{ xsTransposed[i][i] };
			const float dWxhValue{ dhRawValue * xsTValue };
			dWxhVector.push_back(dWxhValue);
		}
		// Might have to be index access instead since all elements are currently 0?
		// Maybe outDeltaWeightsInputToHidden[i] = dWxhVector
		//outDeltaWeightsInputToHidden.push_back(dWxhVector);
		outDeltaWeightsInputToHidden[i] = dWxhVector;

		// dWhh += np.dot(dhraw, hs[t - 1].T)
		std::vector<float> dWhhVector{};
		for (size_t j = 0; j < dhraw.size(); ++j)
		{
			const float dhRawValue{ dhraw[j] };
			const float hsPrevTValue{ hsTransposed[j][i - 1] };
			const float dWxhValue{ dhRawValue * hsPrevTValue };
			dWhhVector.push_back(dWxhValue);
		}
		// Might have to be index access instead since all elements are currently 0?
		// Maybe outDeltaWeightsHiddenToHidden[i] = dWhhVector
		outDeltaWeightsHiddenToHidden[i] = dWhhVector;

		// dhnext = np.dot(Whh.T, dhraw)
		std::vector<float> dhNextVector{};
		for (size_t j = 0; j < whhTransposed.size(); ++j)
		{
			const float whhTValue{ whhTransposed[i][j] };
			const float dhrawValue{ dhraw[j] };
			const float dhNextValue{ whhTValue * dhrawValue };
			dhNextVector.push_back(dhNextValue);
		}
		dhnext = dhNextVector;
	}

	// for dparam in[dWxh, dWhh, dWhy, dbh, dby]:
	//		np.clip(dparam, -5, 5, out = dparam) # clip to mitigate exploding gradients
	const float clampValue{ 1.0f };
	for (size_t i = 0; i < outDeltaWeightsInputToHidden.size(); ++i)
	{
		std::vector<float>& dVector{ outDeltaWeightsInputToHidden[i] };
		for (float& dParam : dVector)
		{
			if (dParam < -clampValue)
			{
				dParam = -clampValue;
			}
			else if (dParam > clampValue)
			{
				dParam = clampValue;
			}
		}
	}
	for (size_t i = 0; i < outDeltaWeightsHiddenToHidden.size(); ++i)
	{
		std::vector<float>& dVector{ outDeltaWeightsHiddenToHidden[i] };
		for (float& dParam : dVector)
		{
			if (dParam < -clampValue)
			{
				dParam = -clampValue;
			}
			else if (dParam > clampValue)
			{
				dParam = clampValue;
			}
		}
	}
	for (size_t i = 0; i < outDeltaWeightsHiddenToOutput.size(); ++i)
	{
		std::vector<float>& dVector{ outDeltaWeightsHiddenToOutput[i] };
		for (float& dParam : dVector)
		{
			if (dParam < -clampValue)
			{
				dParam = -clampValue;
			}
			else if (dParam > clampValue)
			{
				dParam = clampValue;
			}
		}
	}
	for (size_t i = 0; i < outDeltaHiddenBiases.size(); ++i)
	{
		std::vector<float>& dVector{ outDeltaHiddenBiases[i] };
		for (float& dParam : dVector)
		{
			if (dParam < -clampValue)
			{
				dParam = -clampValue;
			}
			else if (dParam > clampValue)
			{
				dParam = clampValue;
			}
		}
	}
	for (size_t i = 0; i < outDeltaOutputBiases.size(); ++i)
	{
		std::vector<float>& dVector{ outDeltaOutputBiases[i] };
		for (float& dParam : dVector)
		{
			if (dParam < -clampValue)
			{
				dParam = -clampValue;
			}
			else if (dParam > clampValue)
			{
				dParam = clampValue;
			}
		}
	}
	hPrev[hPrev.size() - 1] = hs[inputs.size() - 1];
}

void Sample(std::vector<std::vector<float>>& h, int& seed_ix, const size_t numberOfSamples, std::vector<int>& outSampleIDs)
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::vector<std::vector<float>> x{};
	for (size_t i = 0; i < vocabSize; ++i)
	{
		std::vector<float> vec{};
		vec.push_back(0.0f);
		x.push_back(vec);
	}
	for (size_t i = 0; i < x[seed_ix].size(); ++i)
	{
		x[seed_ix][i] = 1.0f;
	}
	outSampleIDs.clear();
	for (size_t i = 0; i < numberOfSamples; ++i)
	{
		// h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
		std::vector<std::vector<float>> dotWxh{};
		for (size_t j = 0; j < weightsInputToHidden.size(); ++j)
		{
			std::vector<float> dotVec{};
			std::vector<float> vec{ weightsInputToHidden[j] };
			for (size_t k = 0; k < vec.size(); ++k)
			{
				const float wxhValue{ vec[k] };
				const float xValue{ x[k][0] };
				const float dotValue{ wxhValue * xValue };
				dotVec.push_back(dotValue);
			}
			dotWxh.push_back(dotVec);
		}
		std::vector<std::vector<float>> dotWhh{};
		for (size_t j = 0; j < weightsHiddenToHidden.size(); ++j)
		{
			std::vector<float> dotVec{};
			std::vector<float> vec{ weightsHiddenToHidden[j] };
			for (size_t k = 0; k < vec.size(); ++k)
			{
				const float whhValue{ vec[k] };
				const float hValue{ h[k][0] };
				const float dotValue{ whhValue * hValue };
				dotVec.push_back(dotValue);
			}
			dotWhh.push_back(dotVec);
		}
		std::vector<std::vector<float>> tanhValues{};
		for (size_t j = 0; j < dotWxh.size(); ++j)
		{
			std::vector<float> dotWxhVec{ dotWxh[j] };
			std::vector<float> dotWhhVec{ dotWhh[j] };
			std::vector<float> tanhVec{};
			for (size_t k = 0; k < dotWxhVec.size(); ++k)
			{
				const float dotWxhVecValue{ dotWxhVec[k] };
				const float dotWhhVecValue{ dotWhhVec[k] };
				const float tanhValue{ std::tanh((dotWxhVecValue * x[k][0]) + (dotWhhVecValue * h[k][0]) + hiddenBiases[k][0]) };
				tanhVec.push_back(tanhValue);
			}
			tanhValues.push_back(tanhVec);
		}
		h = tanhValues;

		// y = np.dot(Why, h) + by
		std::vector<std::vector<float>> y{};
		for (size_t j = 0; j < weightsHiddenToOutput.size(); ++j)
		{
			std::vector<float> vec{ weightsHiddenToOutput[j] };
			std::vector<float> hVec{ h[j] };
			std::vector<float> byVec{ outputBiases[j] };
			std::vector<float> yVec{};

			for (size_t k = 0; k < vec.size(); ++k)
			{
				const float whyValue{ vec[k] };
				const float yValue{ (whyValue * h[k][0]) + byVec[0] };
				yVec.push_back(yValue);
			}
			y.push_back(yVec);
		}

		// p = np.exp(y) / np.sum(np.exp(y))
		std::vector<float> pRavel{};
		std::vector<std::vector<float>> p{};
		std::vector<std::vector<float>> yExp{};
		std::vector<float> yExpSum{};
		for (size_t j = 0; j < y.size(); ++j)
		{
			std::vector<float> yVec{ y[j] };
			std::vector<float> yExpVec{};
			float yExpTotal{ 0.0f };
			for (size_t k = 0; k < yVec.size(); ++k)
			{
				const float yExpValue{ std::exp(yVec[k]) };
				yExpTotal += yExpValue;
				yExpVec.push_back(yExpValue);
			}
			yExpSum.push_back(yExpTotal);
			yExp.push_back(yExpVec);
		}
		for (size_t j = 0; j < yExp.size(); ++j)
		{
			std::vector<float> pVec{};
			std::vector<float> yExpVec{ yExp[j] };
			float yExpTotal{ yExpSum[j] };
			for (size_t k = 0; k < yExpVec.size(); ++k)
			{
				//p = np.exp(y) / np.sum(np.exp(y))
				const float yExpValue{ yExpVec[k] };
				const float y{ yExpValue / yExpTotal };
				pVec.push_back(y);
				pRavel.push_back(y);
			}
			p.push_back(pVec);
		}

		// ix = np.random.choice(range(vocab_size), p = p.ravel())
		// random int from 0 to vocabSize from p.size()
		std::vector<float> randomValues{};
		for (size_t j = 0; j < pRavel.size(); ++j)
		{
			// random values from 0 to vocab size
			std::uniform_real_distribution<float> dis(0.0f, (float)vocabSize);
			const float randomValue{ dis(gen) };
			randomValues.push_back(randomValue);
		}
		std::uniform_real_distribution<float> dis(0.0f, (float)randomValues.size());
		float randomIndexFloat{ dis(gen) };
		int randomIndex{ (int)randomIndexFloat };
		float selectedRandomValue{ randomValues[randomIndex] };
		int ix{ (int)selectedRandomValue };

		// x = np.zeros((vocab_size, 1))
		for (size_t j = 0; j < vocabSize; ++j)
		{
			std::vector<float> vec{};
			vec.push_back(0.0f);
			x[j] = vec;
		}

		// x[ix] = 1
		for (size_t j = 0; j < x[ix].size(); ++j)
		{
			x[ix][j] = 1.0f;
		}

		outSampleIDs.push_back(ix);
	}
}

void Zip(std::vector<std::vector<float>>& vectorA, std::vector<std::vector<float>>& vectorB, std::vector<std::vector<float>>& vectorC,
	std::vector<std::tuple<float&, float&, float&>>& outVector)
{
	for (size_t i = 0; i < vectorA.size(); ++i)
	{
		std::vector<float>& vectorAVec{ vectorA[i] };
		std::vector<float>& vectorBVec{ vectorB[i] };
		std::vector<float>& vectorCVec{ vectorC[i] };

		for (size_t j = 0; j < vectorAVec.size(); ++j)
		{
			float& vecAValue{ vectorAVec[j] };
			float& vecBValue{ vectorBVec.size() == 1 ? vectorBVec[0] : vectorBVec[j] };
			float& vecCValue{ vectorCVec[j] };
			std::tuple<float&, float&, float&> tupleValues = std::tie(vecAValue, vecBValue, vecCValue);
			outVector.push_back(tupleValues);
		}
	}
}

//////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	std::atomic<bool> run(true);
	std::thread asyncInputThread(AsyncInput, std::ref(run));

	const fs::path exeFilepath{ argv[0] };
	//const std::string inputFilepathString{ exeFilepath.parent_path().string() + "\\..\\Assets\\input.txt" };
	const std::string inputFilepathString{ exeFilepath.parent_path().string() + "\\..\\Assets\\shakespeare.txt" };
	const std::string outputFilepathString{ exeFilepath.parent_path().string() + "\\..\\Assets\\output.txt" };
	const fs::path inputFilepath{ inputFilepathString };
	const fs::path outputFilepath{ outputFilepathString };

	// Data I/O
	inputFileText = ReadFile(inputFilepath, false);
	dataSize = inputFileText.size();
	int characterIndex{ 0 };
	for (char character : inputFileText)
	{
		if (charToID.find(character) != charToID.end())
			continue;

		charToID.emplace(character, characterIndex);
		IDToChar.emplace(characterIndex, character);
		++characterIndex;
	}
	vocabSize = charToID.size();
	std::cout << "Data has " << dataSize << " characters and " << vocabSize << " of them are unique" << std::endl;

	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::uniform_real_distribution<float> dis{ 0.0f, 1.0f };
	// Vector of size hiddenSize
		// Each vector then has another vector of length vocabSize
	for (size_t iN = 0; iN < hiddenSize; ++iN)
	{
		std::vector<float> weights{};
		for (size_t iV = 0; iV < vocabSize; ++iV)
		{
			float randomWeightValue{ dis(gen) * 0.01f };
			weights.push_back(randomWeightValue);
		}
		weightsInputToHidden.push_back(weights);
	}
	for (size_t iH = 0; iH < hiddenSize; ++iH)
	{
		std::vector<float> weights{};
		for (size_t iHH = 0; iHH < hiddenSize; ++iHH)
		{
			float randomWeightValue{ dis(gen) * 0.01f };
			weights.push_back(randomWeightValue);
		}
		weightsHiddenToHidden.push_back(weights);
	}
	for (size_t iH = 0; iH < vocabSize; ++iH)
	{
		std::vector<float> weights{};
		for (size_t iHH = 0; iHH < hiddenSize; ++iHH)
		{
			float randomWeightValue{ dis(gen) * 0.01f };
			weights.push_back(randomWeightValue);
		}
		weightsHiddenToOutput.push_back(weights);
	}
	for (size_t iH = 0; iH < hiddenSize; ++iH)
	{
		std::vector<float> biases{};
		for (size_t iB = 0; iB < 1; ++iB)
		{
			biases.push_back(0.0f);
		}
		hiddenBiases.push_back(biases);
	}
	for (size_t iV = 0; iV < vocabSize; ++iV)
	{
		std::vector<float> biases{};
		for (size_t iB = 0; iB < 1; ++iB)
		{
			biases.push_back(0.0f);
		}
		outputBiases.push_back(biases);
	}

	/*
	n, p = 0, 0
	mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
	smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
	*/
	size_t epochs{ 100000 };
	size_t n{ 0 };
	size_t p{ 0 };
	size_t numberOfSamples{ 3 };
	std::vector<std::vector<float>> mWxh{};
	std::vector<std::vector<float>> mWhh{};
	std::vector<std::vector<float>> mWhy{};
	std::vector<std::vector<float>> mbh{};
	std::vector<std::vector<float>> mby{};
	for (size_t i = 0; i < weightsInputToHidden.size(); ++i)
	{
		std::vector<float> mWxhVec{};
		std::vector<float> vec{ weightsInputToHidden[i] };
		for (size_t j = 0; j < vec.size(); ++j)
		{
			mWxhVec.push_back(0.0f);
		}
		mWxh.push_back(mWxhVec);
	}
	for (size_t i = 0; i < weightsHiddenToHidden.size(); ++i)
	{
		std::vector<float> mWhhVec{};
		std::vector<float> vec{ weightsHiddenToHidden[i] };
		for (size_t j = 0; j < vec.size(); ++j)
		{
			mWhhVec.push_back(0.0f);
		}
		mWhh.push_back(mWhhVec);
	}
	for (size_t i = 0; i < weightsHiddenToOutput.size(); ++i)
	{
		std::vector<float> mWhyVec{};
		std::vector<float> vec{ weightsHiddenToOutput[i] };
		for (size_t j = 0; j < vec.size(); ++j)
		{
			mWhyVec.push_back(0.0f);
		}
		mWhy.push_back(mWhyVec);
	}
	for (size_t i = 0; i < hiddenBiases.size(); ++i)
	{
		std::vector<float> mbhVec{};
		std::vector<float> vec{ hiddenBiases[i] };
		for (size_t j = 0; j < vec.size(); ++j)
		{
			mbhVec.push_back(0.0f);
		}
		mbh.push_back(mbhVec);
	}
	for (size_t i = 0; i < outputBiases.size(); ++i)
	{
		std::vector<float> mbyVec{};
		std::vector<float> vec{ outputBiases[i] };
		for (size_t j = 0; j < vec.size(); ++j)
		{
			mbyVec.push_back(0.0f);
		}
		mby.push_back(mbyVec);
	}
	float smoothLoss{ -std::log(1.0f / vocabSize) * sequenceLength };
	std::vector<std::vector<float>> hPrev{};

	std::ofstream outputFile{ outputFilepath };
	if (outputFile.is_open())
	{
		outputFile.clear();

		auto start = std::chrono::steady_clock::now();

		while (run.load())
		{
			/*
			# prepare inputs(we're sweeping from left to right in steps seq_length long)
				if p + seq_length + 1 >= len(data) or n == 0:
					hprev = np.zeros((hidden_size, 1)) # reset RNN memory
					p = 0 # go from start of data
			*/
			if (p + sequenceLength + 1 >= dataSize || n == 0)
			{
				for (size_t i = 0; i < hiddenSize; ++i)
				{
					std::vector<float> vec{};
					vec.push_back(0.0f);
					hPrev.push_back(vec);
				}
				p = 0;
			}

			// inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
			std::vector<int> inputs{};
			for (size_t iC = p; iC < p + sequenceLength; ++iC)
			{
				char character = inputFileText[iC];
				int ID = charToID.at(character);
				inputs.push_back(ID);
			}

			// targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
			std::vector<int> targets{};
			for (size_t iC = p + 1; iC < p + sequenceLength + 1; ++iC)
			{
				char character = inputFileText[iC];
				int ID = charToID.at(character);
				targets.push_back(ID);
			}

			/*
			# sample from the model now and then
			if n % 100 == 0:
			  sample_ix = sample(hprev, inputs[0], 200)
			  txt = ''.join(ix_to_char[ix] for ix in sample_ix)
			  print '----\n %s \n----' % (txt, )
			*/
			if (n > 0 && n % numberOfIterationsToSample == 0)
			{
				std::vector<int> sampleIDs{};
				Sample(hPrev, inputs[0], 200, sampleIDs);

				std::string sampleString{};
#if DEBUG_SAMPLE_TEXT
				sampleString += ("Iteration " + n);
				sampleString += ("\n");
				sampleString += ("Loss " + std::to_string(smoothLoss));
				sampleString += ("\n");
				sampleString += ("===============");
				sampleString += ("\n");
				sampleString += ("Sample Text:");
				sampleString += ("\n");
#endif
				for (size_t iS = 0; iS < sampleIDs.size(); ++iS)
				{
					int sampleIndex{ sampleIDs[iS] };
					const char character = IDToChar.at(sampleIndex);
					std::cout << "----\n" << character << "\n----" << std::endl;
					sampleString += character;
				}
#if DEBUG_SAMPLE_TEXT
				sampleString += ("\n");
				sampleString += ("===============");
#endif

				outputFile << sampleString + "\n";

				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				std::cout << "RNN took " << elapsed_seconds.count() << "s to sample input text file" << std::endl;
				start = std::chrono::steady_clock::now();
			}

			/*
			# forward seq_length characters through the net and fetch gradient
				loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
				smooth_loss = smooth_loss * 0.999 + loss * 0.001
				if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
			*/
			float loss{ 0.0f };
			std::vector<std::vector<float>> dWxh{};
			std::vector<std::vector<float>> dWhh{};
			std::vector<std::vector<float>> dWhy{};
			std::vector<std::vector<float>> dbh{};
			std::vector<std::vector<float>> dby{};
			Loss(inputs, targets, hPrev, loss, dWxh, dWhh, dWhy, dbh, dby);
			smoothLoss = smoothLoss * 0.999f + loss * 0.001f;
			if (n > 0 && n % numberOfIterationsToSample == 0)
			{
				std::cout << "Iter: " << n << ", " << "Loss: " << smoothLoss << std::endl;
			}

			/*
			# perform parameter update with Adagrad
			for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
								[dWxh, dWhh, dWhy, dbh, dby],
								[mWxh, mWhh, mWhy, mbh, mby]):
				mem += dparam * dparam
				param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
			*/
			std::vector<std::tuple<float&, float&, float&>> inputToHiddenParams{};
			std::vector<std::tuple<float&, float&, float&>> hiddenToHiddenParams{};
			std::vector<std::tuple<float&, float&, float&>> hiddenToOutputParams{};
			std::vector<std::tuple<float&, float&, float&>> hiddenBiasParams{};
			std::vector<std::tuple<float&, float&, float&>> outputBiasParams{};
			Zip(weightsInputToHidden, dWxh, mWxh, inputToHiddenParams);
			Zip(weightsHiddenToHidden, dWhh, mWhh, hiddenToHiddenParams);
			Zip(weightsHiddenToOutput, dWhy, mWhy, hiddenToOutputParams);
			Zip(hiddenBiases, dbh, mbh, hiddenBiasParams);
			Zip(outputBiases, dby, mby, outputBiasParams);
			auto func = [&](std::vector<std::tuple<float&, float&, float&>>& tupleVec)
			{
				for (size_t i = 0; i < tupleVec.size(); ++i)
				{
					std::tuple<float&, float&, float&> params{ tupleVec[i] };
					float& param{ std::get<0>(params) };
					float& dparam{ std::get<1>(params) };
					float& mem{ std::get<2>(params) };

					mem += dparam * dparam;
					param += -learningRate * (dparam / std::sqrt(mem + (float)1e-8));
				}
			};
			func(inputToHiddenParams);
			func(hiddenToHiddenParams);
			func(hiddenToOutputParams);
			func(hiddenBiasParams);
			func(outputBiasParams);

			p += sequenceLength;
			++n;

			if (n > 0 && n % epochs == 0)
			{
				run.store(false);
			}
		}

		outputFile.close();
	}

	run.store(false);
	asyncInputThread.join();

	return 0;
}

//////////////////////////////////////////////////////////////////////////