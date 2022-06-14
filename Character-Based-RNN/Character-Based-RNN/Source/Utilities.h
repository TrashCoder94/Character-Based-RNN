#pragma once

#include "Core.h"

class Utilities
{
	public:
		static std::string ReadFile(const fs::path filepath);
		static void Zeros(const size_t a, const size_t b, std::vector<std::vector<float>>& outVec);
		static void Zeros(const size_t a, std::vector<float>& outVec);
		static void InitialiseRandomVectorOfFloatVectors(const size_t a, const size_t b, const float randomMultiplier, std::vector<std::vector<float>>& outVec);
		
		static std::vector<float> CalculateHiddenState(const rnnData& wxh, const std::vector<float>& xs, const rnnData& whh, const std::vector<float>& hs, const rnnData& bh);
		static std::vector<float> CalculateUnormalizedProbabilitiesForNextChars(const rnnData& why, const std::vector<float>& hs, const rnnData& by);
		static std::vector<float> CalculateProbabilitiesForNextChars(const std::vector<float>& ys);
		static std::vector<float> BackPropIntoDeltaH(const rnnData& transposedWhy, const std::vector<float>& dy, const std::vector<float>&  dhNext);
		static std::vector<float> BackPropThroughTanh(const std::vector<float>& hs, const std::vector<float>& dh);
		static void AdagradUpdate(rnnData& wxh, rnnData& dWxh, rnnData& adaWxh
								, rnnData& whh, rnnData& dWhh, rnnData& adaWhh
								, rnnData& why, rnnData& dWhy, rnnData& adaWhy
								, rnnData& bh, rnnData& dbh, rnnData& adaBh
								, rnnData& by, rnnData& dby, rnnData& adaBy
								, const float learningRate);

		static rnnData Dot(const rnnData& a, const rnnData& b);
		static std::vector<float> Dot(const rnnData& a, const std::vector<float>& b);
		static std::vector<float> Dot(const std::vector<float>& a, const std::vector<float>& b);
		
		static std::vector<float> Exp(const std::vector<float>& a);
		static rnnData Exp(const rnnData& a);
		static float Sum(const std::vector<float>& a);
		static float Sum(const rnnData& a);

		static rnnData Transpose(const rnnData& a);

		static void Clip(rnnData& a, float min = -5.0f, float max = 5.0f);

		// h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h) 
		static rnnData GenerateTanh(rnnData& wxh, rnnData& x, rnnData& whh, rnnData& h, rnnData& bh);

		// y = np.dot(W_hy, h) + b_y
		static rnnData GenerateUnormalizedProbabilities(rnnData& why, rnnData& h, rnnData& bh);

		// p = np.exp(y) / np.sum(np.exp(y)) 
		static rnnData GenerateProbabilites(rnnData& y, std::vector<float>& outPRavel);

		static int GenerateRandomIndex(std::vector<float>& pRavel, size_t vocabSize);

		static void Zip(rnnData& a, rnnData& b, rnnData& c,
			std::vector<std::tuple<float&, float&, float&>>& outZippedVector);
};