#pragma once

#include <filesystem>
#include <map>

namespace fs = std::filesystem;

class RecurrentNeuralNetwork
{
	public:
		RecurrentNeuralNetwork(const fs::path& inputFile, const fs::path& outputFile, const fs::path& sequenceFile, bool clean = true);
		~RecurrentNeuralNetwork();

		void ReadFile(const fs::path file);
		void CreateSequences(bool addNewLineAfterEverySequence = false);
		void SaveSequences();
		void CreateDictionary();

	private:
		std::map<char, int> dictionary;

		fs::path inputFilepath;
		fs::path outputFilepath;
		fs::path sequenceFilepath;

		std::string inputString;
		std::string outputString;

		std::string sequenceString;
		size_t sequenceLength;

		bool shouldBeClean;
};