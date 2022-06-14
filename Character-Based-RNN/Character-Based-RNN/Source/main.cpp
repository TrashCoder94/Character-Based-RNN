//////////////////////////////////////////////////////////////////////////

#include "RNN.h"
#include <thread>

//////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	const fs::path exeFilepath{ argv[0] };
	//const std::string inputFilepathString{ exeFilepath.parent_path().string() + "\\..\\Assets\\input.txt" };
	const std::string inputFilepathString{ exeFilepath.parent_path().string() + "\\..\\Assets\\alphabet.txt" };
	const std::string outputFilepathString{ exeFilepath.parent_path().string() + "\\..\\Assets\\output.txt" };
	const fs::path inputFilepath{ inputFilepathString };
	const fs::path outputFilepath{ outputFilepathString };

	const size_t numberOfIterations{ 5000 };
	const size_t sequenceLength{ 10 };
	const size_t hiddenSize{ 100 };
	const float learningRate{ 0.001f };

	RNN rnn{};
	rnn.Init(inputFilepath, numberOfIterations, sequenceLength, hiddenSize, learningRate);
	
	rnn.Training();

	rnn.Generate('b', 25, outputFilepath);
	rnn.Generate('c', 25, outputFilepath);
}

//////////////////////////////////////////////////////////////////////////