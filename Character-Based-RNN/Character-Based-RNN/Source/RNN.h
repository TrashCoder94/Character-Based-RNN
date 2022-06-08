#pragma once

#include <vector>

class RNN
{
	public:
		RNN();
		~RNN();

		void Init(const size_t inputSize, const size_t outputSize, const size_t hiddenSize, const float learningRate);
		float Train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets);

	private:
		size_t m_inputSize;
		size_t m_hiddenSize;
		size_t m_outputSize;
		float m_learningRate;

		std::vector<std::vector<float>> m_h;

		std::vector<std::vector<float>> m_Wxh;
		std::vector<std::vector<float>> m_Whh;
		std::vector<std::vector<float>> m_Why;
		std::vector<std::vector<float>> m_bh;
		std::vector<std::vector<float>> m_by;
		
		std::vector<std::vector<float>> m_adaWxh;
		std::vector<std::vector<float>> m_adaWhh;
		std::vector<std::vector<float>> m_adaWhy;
		std::vector<std::vector<float>> m_adabh;
		std::vector<std::vector<float>> m_adaby;


};