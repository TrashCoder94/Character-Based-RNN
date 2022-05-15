#pragma once

#include "Layer.h"

class HiddenLayer : public Layer
{
	public:
		HiddenLayer();
		explicit HiddenLayer(const size_t numberOfNeurons, const std::vector<float> weightsIn);
		~HiddenLayer();

		virtual void InitLayer() override;

	private:
		std::vector<float> m_weightsIn;
		std::vector<float> m_weightsOut;
};