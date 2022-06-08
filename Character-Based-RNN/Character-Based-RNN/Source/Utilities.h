#pragma once

#include <vector>

class Utilities
{
	public:
		static void Zeros(const size_t a, const size_t b, std::vector<std::vector<float>>& outVec);
		static void InitialiseRandomVector(const size_t a, const size_t b, const float randomMultiplier, std::vector<std::vector<float>>& outVec);
};