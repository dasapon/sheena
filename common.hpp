#pragma once

#include <algorithm>
#include <vector>
#include <sstream>
#include <string>

namespace sheena{
	template<typename Ty, typename F>
	void remove(std::vector<Ty>& v, F f){
		v.erase(std::remove_if(v.begin(),v.end(), f), v.end());
	}
	inline double sigmoid(double d){
		return 1.0 / (1.0 + std::exp(-d));
	}
	inline std::vector<std::string> split_string(const std::string& line, char delim){
		std::vector<std::string> ret;
		std::istringstream iss(line);
		std::string str;
		while(std::getline(iss, str, delim))ret.push_back(str);
		return ret;
	}
}