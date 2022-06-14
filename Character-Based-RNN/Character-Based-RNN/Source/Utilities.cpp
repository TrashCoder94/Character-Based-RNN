#include "Utilities.h"
#include <fstream>
#include <random>
#include <sstream>


std::string Utilities::ReadFile(const fs::path filepath)
{
	std::string fileText{ "" };

	// Open input file
	std::ifstream inputFile{ filepath };
	if (inputFile.is_open())
	{
		std::string line{ "" };

		// Read the input file line by line
		while (std::getline(inputFile, line))
		{
#if 0
			// If we want this to be "clean" then make sure there are no new lines
			// This will create one long sequence of strings
			const std::string stringToAppend = addNewLineAtEnd ? line + "\n" : line;
			fileText.append(stringToAppend);
#else
			fileText.append(line);
#endif
		}
		inputFile.close();
	}

	return fileText;
}

void Utilities::Zeros(const size_t a, const size_t b, std::vector<std::vector<float>>& outVec)
{
	outVec.clear();

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

void Utilities::Zeros(const size_t a, std::vector<float>& outVec)
{
	outVec.clear();

	for (size_t i = 0; i < a; ++i)
	{
		outVec.push_back(0.0f);
	}
}

void Utilities::InitialiseRandomVectorOfFloatVectors(const size_t a, const size_t b, const float randomMultiplier, std::vector<std::vector<float>>& outVec)
{
	outVec.clear();

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

std::vector<float> Utilities::CalculateHiddenState(const rnnData& wxh, const std::vector<float>& xs, const rnnData& whh, const std::vector<float>& hs, const rnnData& bh)
{
	//  hs[t] = np.tanh(np.dot(W_xh, xs[t]) + np.dot(W_hh, hs[t-1]) + b_h) # hidden state.
	std::vector<float> tanhData{};
	std::vector<float> wxhDotXs{ Dot(wxh, xs) };
	std::vector<float> whhDotHs{ Dot(whh, hs) };
	for (size_t i = 0; i < wxhDotXs.size(); ++i)
	{
		const float wxhDotXsVal{ wxhDotXs[i] };
		const float whhDotHsVal{ whhDotHs[i] };
		const float bias{ (i > bh.size() - 1) ? 0.0f : bh[i][0] };
		const float tanhVal{ std::tanh(wxhDotXsVal + whhDotHsVal + bias) };
		tanhData.push_back(tanhVal);
	}
	return tanhData;
}

std::vector<float> Utilities::CalculateUnormalizedProbabilitiesForNextChars(const rnnData& why, const std::vector<float>& hs, const rnnData& by)
{
	// ys[t] = np.dot(W_hy, hs[t]) + b_y # unnormalized log probabilities for next chars
	std::vector<float> probs{};
	std::vector<float> whyDotHs{ Dot(why, hs) };
	for (size_t i = 0; i < whyDotHs.size(); ++i)
	{
		const float whyDotHsVal{ whyDotHs[i] };
		const float bias{ (i > by.size() - 1) ? 0.0f : (by[i][0]) };
		const float probability{ whyDotHsVal + bias };
		probs.push_back(probability);
	}
	return probs;
}

std::vector<float> Utilities::CalculateProbabilitiesForNextChars(const std::vector<float>& ys)
{
	// ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars. 
	std::vector<float> probabilities{  };
	std::vector<float> exp{ Exp(ys) };
	const float sum{ Sum(exp) };

	for (size_t i = 0; i < exp.size(); ++i)
	{
		const float val{ exp[i] };
		const float probability{ val / sum };
		probabilities.push_back(probability);
	}

	return probabilities;
}

std::vector<float> Utilities::BackPropIntoDeltaH(const rnnData& transposedWhy, const std::vector<float>& dy, const std::vector<float>& dhNext)
{
	// dh = np.dot(W_hy.T, dy) + dhnext # backprop into h.
	std::vector<float> backPropIntoH{};
	std::vector<float> dot{ Dot(transposedWhy, dy) };
	
	for (size_t i = 0; i < dot.size(); ++i)
	{
		const float dotVal{ dot[i] };
		//const float dhNextVal{ (i > dhNext.size() - 1) ? 0.0f : dhNext[i] };
		const float dhNextVal{ dhNext[i] };
		const float backPropIntoHVal{ dotVal + dhNextVal };
		backPropIntoH.push_back(backPropIntoHVal);
	}
	return backPropIntoH;
}

std::vector<float> Utilities::BackPropThroughTanh(const std::vector<float>& hs, const std::vector<float>& dh)
{
	// dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
	std::vector<float> dhRaw{};

	for (size_t i = 0; i < hs.size(); ++i)
	{
		/*const float hsVal{ i > hs.size() - 1 ? 0.0f : hs[i] };
		const float dhVal{ i > dh.size() - 1 ? 0.0f: dh[i] };*/
		const float hsVal{ hs[i] };
		const float dhVal{ dh[i] };
		const float dhRawVal{ (1.0f - (hsVal * hsVal)) * dhVal };
		dhRaw.push_back(dhRawVal);
	}

	return dhRaw;
}

void Utilities::AdagradUpdate(rnnData& wxh, rnnData& dWxh, rnnData& adaWxh, rnnData& whh, rnnData& dWhh, rnnData& adaWhh, rnnData& why, rnnData& dWhy, rnnData& adaWhy, rnnData& bh, rnnData& dbh, rnnData& adaBh, rnnData& by, rnnData& dby, rnnData& adaBy, const float learningRate)
{
	std::vector<std::tuple<float&, float&, float&>> inputToHiddenParams{};
	std::vector<std::tuple<float&, float&, float&>> hiddenToHiddenParams{};
	std::vector<std::tuple<float&, float&, float&>> hiddenToOutputParams{};
	std::vector<std::tuple<float&, float&, float&>> hiddenBiasParams{};
	std::vector<std::tuple<float&, float&, float&>> outputBiasParams{};
	Utilities::Zip(wxh, dWxh, adaWxh, inputToHiddenParams);
	Utilities::Zip(whh, dWhh, adaWhh, hiddenToHiddenParams);
	Utilities::Zip(why, dWhy, adaWhy, hiddenToOutputParams);
	Utilities::Zip(bh, dbh, adaBh, hiddenBiasParams);
	Utilities::Zip(by, dby, adaBy, outputBiasParams);

	auto func = [&](std::vector<std::tuple<float&, float&, float&>>& tupleVec)
	{
		for (size_t i = 0; i < tupleVec.size(); ++i)
		{
			std::tuple<float&, float&, float&> params{ tupleVec[i] };
			float& param{ std::get<0>(params) };
			float& dparam{ std::get<1>(params) };
			float& mem{ std::get<2>(params) };

			mem += (dparam * dparam);
			param += ((-learningRate * dparam) / std::sqrt(mem + (float)1e-8));
		}
	};
	func(inputToHiddenParams);
	func(hiddenToHiddenParams);
	func(hiddenToOutputParams);
	func(hiddenBiasParams);
	func(outputBiasParams);
}

rnnData Utilities::Dot(const rnnData& a, const rnnData& b)
{
	rnnData dotData{};

	for (size_t i = 0; i < a.size(); ++i)
	{
		std::vector<float> vecA = a[i];
		std::vector<float> vecB = ((i > b.size() - 1) ? std::vector<float>(1, 0.0f) : b[i]);
		std::vector<float> dotVec{};

		for (size_t j = 0; j < vecA.size(); ++j)
		{
			const float aVal{ vecA[j] };
			const float bVal{ (j > vecB.size() - 1) ? vecB[0] : vecB[j] }; // TODO: vecB may have a size of 1...
			const float dotVal{ aVal * bVal };
			dotVec.push_back(dotVal);
		}

		dotData.push_back(dotVec);
	}

	return dotData;
}

#define DOT_RNNDATA_LOOP 0

std::vector<float> Utilities::Dot(const rnnData& a, const std::vector<float>& b)
{
	std::vector<float> dotVec{};

#if DOT_RNNDATA_LOOP
	for (size_t i = 0; i < a.size(); ++i)
	{
		const std::vector<float>& aVec{ a[i] };
		for (size_t j = 0; j < aVec.size(); ++j)
		{
			const float aVal{ aVec[j] };
			const float bVal{ i > b.size() - 1 ? 0.0f : b[i] }; // maybe b[i]?
			const float dotVal{ aVal * bVal };
			dotVec.push_back(dotVal);
		}
	}
#else
	const std::vector<float>& aVec{ a[/*a.size() - 1*/0] };
	for (size_t i = 0; i < aVec.size(); ++i)
	{
		const float aVal{ i > aVec.size() - 1 ? 0.0f : aVec[i] };
		const float bVal{ i > b.size() - 1 ? 0.0f : b[i] };
		const float dotVal{ aVal * bVal };
		dotVec.push_back(dotVal);
	}
#endif

	return dotVec;
}

std::vector<float> Utilities::Dot(const std::vector<float>& a, const std::vector<float>& b)
{
	std::vector<float> dotVec{};

	for (size_t i = 0; i < a.size(); ++i)
	{
		const float aVal{ i > a.size() - 1 ? 0.0f : a[i] };
		const float bVal{ i > b.size() - 1 ? 0.0f : b[i] };
		const float dotVal{ aVal * bVal };
		dotVec.push_back(dotVal);
	}

	return dotVec;
}

std::vector<float> Utilities::Exp(const std::vector<float>& a)
{
	std::vector<float> expData{};

	for (size_t i = 0; i < a.size(); ++i)
	{
		const float expVal{ std::exp(a[i]) };
		expData.push_back(expVal);
	}

	return expData;
}

rnnData Utilities::Exp(const rnnData& a)
{
	rnnData expData{};

	for (size_t i = 0; i < a.size(); ++i)
	{
		std::vector<float> expVec{};
		const std::vector<float> aVec{ a[i] };

		for (size_t j = 0; j < aVec.size(); ++j)
		{
			const float val{ aVec[j] };
			const float expVal{ std::exp(val) };
			expVec.push_back(expVal);
		}

		expData.push_back(expVec);
	}

	return expData;
}

float Utilities::Sum(const std::vector<float>& a)
{
	float sum{ 0.0f };
	for (size_t i = 0; i < a.size(); ++i)
	{
		const float val{ a[i] };
		sum += val;
	}
	return sum;
}

float Utilities::Sum(const rnnData& a)
{
	float sum{ 0.0f };
	for (size_t i = 0; i < a.size(); ++i)
	{
		const std::vector<float> aVec{ a[i] };

		for (size_t j = 0; j < aVec.size(); ++j)
		{
			const float val{ aVec[j] };
			sum += val;
		}
	}
	return sum;
}

rnnData Utilities::Transpose(const rnnData& a)
{
	size_t startSize{ a[0].size() };
	rnnData transposed = std::vector(startSize, std::vector<float>());

	for (size_t i = 0; i < a.size(); ++i)
	{
		for (size_t j = 0; j < a[i].size(); ++j)
		{
			transposed[j].push_back(a[i][j]);
		}
	}

	return transposed;
}

void Utilities::Clip(rnnData& a, float min /*= -5.0f*/, float max /*= 5.0f*/)
{
	for (size_t i = 0; i < a.size(); ++i)
	{
		std::vector<float>& aVec{ a[i] };
		for (size_t j = 0; j < aVec.size(); ++j)
		{
			float& aVecValue{ aVec[j] };
			if (aVecValue < min)
				aVecValue = min;
			else if (aVecValue > max)
				aVecValue = max;
		}
	}
}

rnnData Utilities::GenerateTanh(rnnData& wxh, rnnData& x, rnnData& whh, rnnData& h, rnnData& bh)
{
	rnnData tanh{};

	// // h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h) 
	rnnData dotA{ Dot(wxh, x) };
	rnnData dotB{ Dot(whh, h) };

	for (size_t i = 0; i < dotA.size(); ++i)
	{
		std::vector<float> tanhVec{};
		const std::vector<float>& aVec{ dotA[i] };
		const std::vector<float>& bVec{ dotB[i] };
		const std::vector<float>& cVec{ bh[i] };

		for (size_t j = 0; j < aVec.size(); ++j)
		{
			const float aVal{ aVec[j] };
			const float bVal{ bVec[j] };
			const float bias{ cVec[0] };
			const float tanhVal{ std::tanh(aVal + bVal + bias) };
			tanhVec.push_back(tanhVal);
		}

		tanh.push_back(tanhVec);
	}

	return tanh;
}

rnnData Utilities::GenerateUnormalizedProbabilities(rnnData& why, rnnData& h, rnnData& bh)
{
	rnnData output{};

	// // y = np.dot(W_hy, h) + b_y
	rnnData dot{ Dot(why, h) };
	for (size_t i = 0; i < dot.size(); ++i)
	{
		std::vector<float> outputVec{};
		const std::vector<float>& dotVec{ dot[i] };
		const std::vector<float>& bhVec{ bh[i] };

		for (size_t j = 0; j < dotVec.size(); ++j)
		{
			const float dotVal{ dotVec[j] };
			const float bias{ bhVec[0] };
			const float outputVal{ dotVal + bias };
			outputVec.push_back(outputVal);
		}

		output.push_back(outputVec);
	}

	return output;
}

rnnData Utilities::GenerateProbabilites(rnnData& y, std::vector<float>& outPRavel)
{
	rnnData p{};

	// p = np.exp(y) / np.sum(np.exp(y)) 
	rnnData exp{ Exp(y) };
	const float sum{ Sum(exp) };

	for (size_t i = 0; i < exp.size(); ++i)
	{
		std::vector<float> pVec{};
		const std::vector<float> expVec{ exp[i] };

		for (size_t j = 0; j < expVec.size(); ++j)
		{
			const float expVal{ expVec[j] };
			const float pVal{ expVal / sum };
			pVec.push_back(pVal);
			outPRavel.push_back(pVal);
		}

		p.push_back(pVec);
	}

	return p;
}

int Utilities::GenerateRandomIndex(std::vector<float>& pRavel, size_t vocabSize)
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };

	// ix = np.random.choice(range(vocab_size), p = p.ravel())
	// random int from 0 to vocabSize from p.size()
	std::vector<float> randomValues{};
	for (size_t j = 0; j < pRavel.size(); ++j)
	{
		// random values from 0 to vocab size
		std::uniform_real_distribution<float> dis(0.0f, (float)vocabSize);
		const float randomValue{ dis(gen) };
		randomValues.push_back(randomValue);
	}
	std::uniform_real_distribution<float> dis(0.0f, (float)randomValues.size());
	float randomIndexFloat{ dis(gen) };
	int randomIndex{ (int)randomIndexFloat };
	float selectedRandomValue{ randomValues[randomIndex] };
	int ix{ (int)selectedRandomValue };
	return ix;
}

void Utilities::Zip(rnnData& a, rnnData& b, rnnData& c, std::vector<std::tuple<float&, float&, float&>>& outZippedVector)
{
	for (size_t i = 0; i < a.size(); ++i)
	{
		std::vector<float>& vectorAVec{ a[i] };
		std::vector<float>& vectorBVec{ b[i] };
		std::vector<float>& vectorCVec{ c[i] };

		for (size_t j = 0; j < vectorAVec.size(); ++j)
		{
			float& vecAValue{ vectorAVec[j] };
			float& vecBValue{ vectorBVec.size() == 1 ? vectorBVec[0] : vectorBVec[j] };
			float& vecCValue{ vectorCVec[j] };
			std::tuple<float&, float&, float&> tupleValues = std::tie(vecAValue, vecBValue, vecCValue);
			outZippedVector.push_back(tupleValues);
		}
	}

	//for (size_t i = 0; i < a.size(); ++i)
	//{
	//	std::vector<float>& aVec{ a[i] };
	//	std::vector<float>& bVec{ b[i] };
	//	std::vector<float>& cVec{ c[i] };
	//	/*size_t maxSize{ aVec.size() };
	//	if (maxSize > bVec.size())
	//	{
	//		maxSize = bVec.size();

	//		if (bVec.size() > cVec.size())
	//		{
	//			maxSize = cVec.size();
	//		}
	//	}*/

	//	for (size_t j = 0; j < aVec.size(); ++j)
	//	{
	//		float& vecAValue{ aVec[j] };
	//		float& vecBValue{ j > bVec.size() - 1 ? bVec[bVec.size() - 1] : bVec[j] };
	//		float& vecCValue{ cVec[j] };
	//		std::tuple<float&, float&, float&> tupleValues = std::tie(vecAValue, vecBValue, vecCValue);
	//		outZippedVector.push_back(tupleValues);
	//	}
	//}
}