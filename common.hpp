#pragma once

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

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
			for(size_t i=0;i<Size && i < init.size();i++){
				array_[i] = *itr;
				itr++;
			}
		}
		Array(const Array<Ty, Size>& base){
			(*this) = base;
		}
		static size_t size(){return Size;}
		const Ty& operator[](size_t idx)const{
			assert(idx < Size);
			return array_[idx];
		}
		Ty& operator[](size_t idx){
			assert(idx < Size);
			return array_[idx];
		}
		void operator=(const Array<Ty, Size> & a){
			for(size_t i=0;i<Size;i++)(*this)[i] = a[i];
		}
		const Ty* begin()const{return array_;}
		const Ty* end()const{return array_ + Size;}
		Ty* begin(){return array_;}
		Ty* end(){return array_ + Size;}
	};
	template<typename Ty, size_t Sz1, size_t Sz2>
	using Array2d = Array<Array<Ty, Sz2>, Sz1>;
	template<typename Ty, size_t Sz1, size_t Sz2, size_t Sz3>
	using Array3d = Array<Array<Array<Ty, Sz3>, Sz2>, Sz1>;

	template<typename Ty>
	class ArrayAlloc{
		Ty* array_;
		size_t size_;
	public:
		ArrayAlloc():array_(nullptr), size_(0){}
		ArrayAlloc(size_t sz):array_(nullptr), size_(0){
			resize(sz);
		}
		ArrayAlloc(const ArrayAlloc<Ty>& base):array_(nullptr), size_(0){
			(*this) = base;
		}
		~ArrayAlloc(){
			if(array_ != nullptr)_mm_free(reinterpret_cast<void*>(array_));
		}
		void resize(size_t sz){
			resize(sz, 16);
		}
		void resize(size_t sz, size_t align){
			if(sz == 0)throw std::invalid_argument("array size is zero");
			if(array_ != nullptr)_mm_free(reinterpret_cast<void*>(array_));
			array_ = reinterpret_cast<Ty*>(_mm_malloc(sizeof(Ty) * sz, align));
			if(array_ == nullptr)throw std::bad_alloc();
			size_ = sz;
		}
		size_t size()const{return size_;}

		const Ty& operator[](size_t idx)const{
			assert(idx >= 0);
			assert(idx < size_);
			return array_[idx];
		}
		Ty& operator[](size_t idx){
			assert(idx >= 0);
			assert(idx < size_);
			return array_[idx];
		}
		void operator=(const ArrayAlloc<Ty> & a){
			if(a.size() != size_)resize(a.size());
			for(size_t i=0;i<size_;i++)(*this)[i] = a[i];
		}
		const Ty* begin()const{return array_;}
		const Ty* end()const{return array_ + size_;}
		Ty* begin(){return array_;}
		Ty* end(){return array_ + size_;}
	};
}