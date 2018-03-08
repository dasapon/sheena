#pragma once

#include <algorithm>
#include <vector>

namespace sheena{
	template<typename Ty, typename F>
	void remove(std::vector<Ty>& v, F f){
		v.erase(std::remove_if(v.begin(),v.end(), f), v.end());
	}
	inline double sigmoid(double d){
		return 1.0 / (1.0 + std::exp(-d));
	}
}