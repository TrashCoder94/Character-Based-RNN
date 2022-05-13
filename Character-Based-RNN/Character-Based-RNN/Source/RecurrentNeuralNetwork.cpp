#include "RecurrentNeuralNetwork.h"
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>

RecurrentNeuralNetwork::RecurrentNeuralNetwork(const fs::path& inputFile, const fs::path& outputFile, const fs::path& sequenceFile, bool clean /*= true*/) :
	inputFilepath{ inputFile },
	outputFilepath{ outputFile },
	sequenceFilepath{ sequenceFile },
	inputString{""},
	outputString{""},
	sequenceLength{ 10 },
	shouldBeClean{ clean }
{}

RecurrentNeuralNetwork::~RecurrentNeuralNetwork()
{}

void RecurrentNeuralNetwork::ReadFile(const fs::path file)
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

void RecurrentNeuralNetwork::CreateSequences(bool addNewLineAfterEverySequence /* = false */)
{
	sequenceString.clear();

	for (size_t iC = sequenceLength; iC < inputString.size(); ++iC)
	{
		std::string subSequenceString{ inputString.substr(iC - sequenceLength, sequenceLength + 1) };
		if (addNewLineAfterEverySequence)	subSequenceString += "\n";
		sequenceString.append(subSequenceString);
	}

	std::cout << "Total sequences " << sequenceString.size() << std::endl;
}

void RecurrentNeuralNetwork::SaveSequences()
{
	// Create the directory for the sequence file
	std::filesystem::create_directories(sequenceFilepath.parent_path());

	std::ofstream sequenceFile{ sequenceFilepath };
	if (sequenceFile.is_open())
	{
		std::cout << "Creating sequence file " << sequenceFilepath << std::endl;

		// Place sequenceString into the sequenceFile.
		sequenceFile.clear();
		sequenceFile << sequenceString;
		sequenceFile.close();
	}
}

void RecurrentNeuralNetwork::CreateDictionary()
{
	int characterIndex = 0;

	for (char character : inputString)
	{
		// If we have already stored this character in our dictionary
		// Continue on to the next character.
		if (dictionary.find(character) != dictionary.end())
			continue;

		dictionary.emplace(character, characterIndex);
		++characterIndex;
	}

	std::cout << "Vocab size " << dictionary.size() << std::endl;
}