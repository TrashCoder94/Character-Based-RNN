#include "Utilities.h"
#include <random>

void Utilities::Zeros(const size_t a, const size_t b, std::vector<std::vector<float>>& outVec)
{
	for (size_t i = 0; i < a; ++i)
	{
		std::vector<float> vec{};
		for (size_t j = 0; j < b; ++j)
		{
			vec.push_back(0.0f);
		}
		outVec.push_back(vec);
	}
}

void Utilities::InitialiseRandomVector(const size_t a, const size_t b, const float randomMultiplier, std::vector<std::vector<float>>& outVec)
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::uniform_real_distribution<float> dis{ 0.0f, 1.0f };

	for (size_t i = 0; i < a; ++i)
	{
		std::vector<float> vec{};
		for (size_t j = 0; j < b; ++j)
		{
			float randomValue{ dis(gen) * randomMultiplier };
			vec.push_back(randomValue);
		}
		outVec.push_back(vec);
	}
}
