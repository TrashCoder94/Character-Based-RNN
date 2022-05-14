#pragma once

#include <filesystem>
#include <map>
#include <vector>

namespace fs = std::filesystem;

class RecurrentNeuralNetwork
{
	public:
		RecurrentNeuralNetwork(const fs::path& inputFile, const fs::path& outputFile);
		~RecurrentNeuralNetwork();

		void ReadFile(const fs::path file, bool shouldBeClean = false);
		void CreateDictionary();

		void Loss(std::vector<int>& inputs, std::vector<int>& target, float& outLoss, 
					std::vector<float>& outDeltaInputToHiddenWeights, std::vector<float>& outDeltaHiddenToHiddenWeights,
					std::vector<float>& outDeltaHiddenToOutputWeights, float& outDeltaHiddenBias, float& outDeltaOutputBias,
					float& outHiddenState);

		void Sample(float& h, const size_t& seed_ix, const size_t numberOfSamples, std::vector<int>& outSamples);
		void Process();

		float Dot(const float& a, const float& b);

	private:
		std::map<char, int> dictionaryCharToID;
		std::map<int, char> dictionaryIDToChar;

		fs::path inputFilepath;
		fs::path outputFilepath;

		std::string inputString;
		std::string outputString;

		size_t hiddenLayerSize; // Number of neurons in the hidden layer
		size_t sequenceLength; // How many times to unroll the RNN for
		
		float learningRate;

		float inputToHiddenWeight;
		float hiddenToHiddenWeight;
		float hiddenToOutputWeight;

		float hiddenBias;
		float outputBias;

		int cycles;

		bool isRunning;
};