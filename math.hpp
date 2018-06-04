#include <algorithm>
#include <float.h>
#include <cmath>

#include "common.hpp"

namespace sheena{
	inline double sigmoid(double d){
		return 1.0 / (1.0 + std::exp(-d));
	}
	template<int Sz>
	inline void softmax(Array<float, Sz>& scores, int n){
		float max_score = -FLT_MAX, sum = 0;
		for(int i=0;i<n;i++){
			max_score = std::max(scores[i], max_score);
		}
		for(int i=0;i<n;i++){
			scores[i] = std::exp(scores[i] - max_score);
			sum += scores[i];
		}
		for(int i=0;i<n;i++){
			scores[i] /= sum;
		}
	}
}