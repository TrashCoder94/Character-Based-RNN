#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "RecurrentNeuralNetwork.h"

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
	std::string inputFilepathString{ "" };
	std::string outputFilepathString{ "" };
	std::string sequenceFilepathString{ "" };

	// If there have been no arguments provided, just use the same input/output text files.
	if (argc != 4)
	{
		std::cout << "Not enough command arguments given, so this will end up using the sample input/output files. Command should be something like this instead: Charcter-Based-RNN.exe pathToInput/input.txt pathToOutput/output.txt" << std::endl;

		const fs::path exeFilepath{ argv[0] };
		inputFilepathString = exeFilepath.parent_path().string() + "\\..\\Assets\\input.txt";
		outputFilepathString = exeFilepath.parent_path().string() + "\\..\\Assets\\output.txt";
		sequenceFilepathString = exeFilepath.parent_path().string() + "\\..\\Assets\\sequence.txt";
	}
	else
	{
		inputFilepathString = argv[1];
		outputFilepathString = argv[2];
		sequenceFilepathString = argv[3];
	}

	const fs::path inputFilepath{ inputFilepathString };
	const fs::path outputFilepath{ outputFilepathString };
	const fs::path sequenceFilepath{ sequenceFilepathString };

	// Create some data to train our neural network with
	RecurrentNeuralNetwork rnn{ inputFilepath, outputFilepath, sequenceFilepath };
	rnn.ReadFile(inputFilepath);
	rnn.CreateSequences();
	rnn.SaveSequences();

	// Train our network now.
	rnn.ReadFile(sequenceFilepath);


	return 0;
}