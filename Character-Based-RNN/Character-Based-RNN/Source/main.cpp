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
	const size_t numberOfInputNeurons{ 2 };
	const size_t numberOfHiddenNeurons{ 2 };
	const size_t numberOfOutputNeurons{ 1 };
	const size_t numberOfRecurrentIterations{ 3 };
	const std::vector<float> inputTest{ 2.0f, 3.0f };

	NeuralNetwork nn{ inputTest, numberOfInputNeurons, numberOfHiddenNeurons, numberOfOutputNeurons };
	nn.Init();
	//const float nnFeedForward = nn.FeedForward(inputTest);
	//std::cout << "NN Feed Forward = " << nnFeedForward << std::endl;

	std::cout << "NN Recurrent" << std::endl;
	const std::vector<float> nnRecurrent = nn.Recurrent(inputTest, numberOfRecurrentIterations);
	for (size_t iR = 0; iR < nnRecurrent.size(); ++iR)
		std::cout << "Recurrent Value #" << iR << " = " << nnRecurrent[iR] << std::endl;
#endif

	return 0;
}