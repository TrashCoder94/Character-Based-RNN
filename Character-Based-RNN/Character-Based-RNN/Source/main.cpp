#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

#define RNN_ATTEMPT 0

#if RNN_ATTEMPT
#include "RecurrentNeuralNetwork.h"
#else
#include "NeuralNetworks/NeuralNetwork.h"
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
	std::string inputFilepathString{ "" };
	std::string outputFilepathString{ "" };

	// If there have been no arguments provided, just use the same input/output text files.
	if (argc != 3)
	{
		std::cout << "Not enough command arguments given, so this will end up using the sample input/output files. Command should be something like this instead: Charcter-Based-RNN.exe pathToInput/input.txt pathToOutput/output.txt" << std::endl;

		const fs::path exeFilepath{ argv[0] };
		inputFilepathString = exeFilepath.parent_path().string() + "\\..\\Assets\\input.txt";
		outputFilepathString = exeFilepath.parent_path().string() + "\\..\\Assets\\output.txt";
	}
	else
	{
		inputFilepathString = argv[1];
		outputFilepathString = argv[2];
	}

	const fs::path inputFilepath{ inputFilepathString };
	const fs::path outputFilepath{ outputFilepathString };

#if RNN_ATTEMPT
	// Create some data to train our neural network with
	RecurrentNeuralNetwork rnn{ inputFilepath, outputFilepath };
	rnn.ReadFile(inputFilepath);
	rnn.CreateDictionary();
	rnn.Process();
#else
	const size_t numberOfInputNeurons{ 5 };
	const size_t numberOfHiddenNeurons{ 5 };
	const size_t numberOfHiddenLayers{ 3 };
	const size_t numberOfOutputNeurons{ 5 };

	NeuralNetwork nn{ numberOfInputNeurons, numberOfHiddenNeurons, numberOfHiddenLayers, numberOfOutputNeurons };
	nn.Init();
	nn.Process();
	nn.Print();

#endif

	return 0;
}