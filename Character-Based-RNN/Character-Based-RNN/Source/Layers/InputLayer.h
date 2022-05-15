#pragma once

#include "Layer.h"

class InputLayer : public Layer
{
	public:
		InputLayer();
		explicit InputLayer(const size_t numberOfNeurons);
		~InputLayer();

		virtual void InitLayer() override;
};