#pragma once

#include "Core.h"

class RNN
{
	public:
		RNN();
		~RNN();
		void LoadTextData(const fs::path dataFilepath);
		void OneHotEncode();
		
		void Init(const fs::path dataFilepath, const size_t iterations, const size_t sequenceLength, const size_t hiddenSize, const float learningRate);
		
		float ForwardPropagation(std::vector<int>& inputs, std::vector<int>& targets, rnnData& hPrev,
			rnnData& outPs, rnnData& outHs, rnnData& outXs);
		
		void BackwardPropagation(std::vector<int>& inputs, std::vector<int>& targets, rnnData& ps, rnnData& xs, rnnData& hs,
			rnnData& outDWxh, rnnData& outDWhh, rnnData& outDWhy, rnnData& outDbh, rnnData& outDby);

		void Training();

		void Generate(char character, size_t lengthOfWordToGenerate, fs::path outputFilepath);

	private:
		std::string m_textData;
		size_t m_textDataSize;
		size_t m_vocabSize;

		std::map<char, int> m_charToID;
		std::map<int, char> m_IDToChar;

		size_t m_numberOfIterations;
		size_t m_sequenceLength;
		size_t m_batchSize;
		size_t m_hiddenSize;
		float m_learningRate;

		rnnData m_Wxh;
		rnnData m_Whh;
		rnnData m_Why;
		rnnData m_bh;
		rnnData m_by;
		rnnData m_hPrev;

		rnnData m_adaWxh;
		rnnData m_adaWhh;
		rnnData m_adaWhy;
		rnnData m_adaBh;
		rnnData m_adaBy;
};