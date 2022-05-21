#pragma once

#include "Layer.h"

class OutputLayer : public Layer
{
	public:
		OutputLayer();
		explicit OutputLayer(const size_t numberOfNeurons, const std::vector<float> inputValues, const std::vector<float>& weightsIn);
		~OutputLayer();

		virtual void InitLayer() override;

	private:
		std::vector<float> m_inputValues;
		std::vector<float> m_weightsIn;
};