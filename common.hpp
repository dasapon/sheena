#pragma once

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <vector>

namespace sheena{
	template<typename Ty, typename F>
	void remove(std::vector<Ty>& v, F f){
		v.erase(std::remove_if(v.begin(),v.end(), f), v.end());
	}
	inline std::vector<std::string> split_string(const std::string& line, char delim){
		std::vector<std::string> ret;
		std::istringstream iss(line);
		std::string str;
		while(std::getline(iss, str, delim))ret.push_back(str);
		return ret;
	}
	
	template<typename Ty, size_t Size>
	class Array{
		Ty array_[Size];
	public:
		Array(){}
		explicit Array(const std::initializer_list<Ty> init){
			auto itr = init.begin();
			for(int i=0;i<Size && i < init.size();i++){
				array_[i] = *itr;
				itr++;
			}
		}
		static size_t size(){return Size;}
		Ty operator[](int idx)const{
			assert(idx >= 0);
			assert(idx < Size);
			return array_[idx];
		}
		Ty& operator[](int idx){
			assert(idx >= 0);
			assert(idx < Size);
			return array_[idx];
		}
		void operator=(const Array<Ty, Size> & a){
			for(int i=0;i<Size;i++)(*this)[i] = a[i];
		}
	};
}