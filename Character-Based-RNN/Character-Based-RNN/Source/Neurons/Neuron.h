#pragma once

#include <vector>

class Neuron
{
	public:
		Neuron();
		explicit Neuron(const std::vector<float> weightsIn);
		~Neuron();

		float Init();
		const void Print() const;
		const float GetSum() const;

	private:
		std::vector<float> m_weightsIn;
		float m_outputValue;
		float m_error;
};