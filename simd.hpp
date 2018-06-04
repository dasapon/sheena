#pragma once

#ifdef __FMA__
#define FMA_ENABLE
#endif
#if defined(__AVX__) && defined(__AVX2__)
#define SIMD256_ENABLE
#include <nmmintrin.h>
#endif
#include <x86intrin.h>

namespace sheena{
	class Float4 {
	union {
		float w[4];
		__m128 m128;
	};
	Float4(__m128 m):m128(m){}
public:
	Float4(float f):m128(_mm_set1_ps(f)){}
	Float4():m128(_mm_setzero_ps()){}
	Float4(const Float4& f4):m128(f4.m128){}
	Float4(float w0, float w1,float w2, float w3){
		w[0] = w0;
		w[1] = w1;
		w[2] = w2;
		w[3] = w3;
	}
	void clear(){
		m128 = _mm_setzero_ps();
	}
	void operator=(const Float4& f) { m128 = f.m128; }

	Float4 operator+(const Float4& f)const { return _mm_add_ps(m128, f.m128); }
	Float4 operator-(const Float4& f)const { return _mm_sub_ps(m128, f.m128); }
	Float4 operator*(const Float4& f)const { return _mm_mul_ps(m128, f.m128); }
	Float4 operator/(const Float4& f)const { return _mm_div_ps(m128, f.m128); }

	void operator+=(const Float4& f) { m128 = _mm_add_ps(m128, f.m128); }
	void operator-=(const Float4& f) { m128 = _mm_sub_ps(m128, f.m128); }
	void operator*=(const Float4& f) { m128 = _mm_mul_ps(m128, f.m128); }
	void operator/=(const Float4& f) { m128 = _mm_div_ps(m128, f.m128); }

	float sum()const { return w[0] + w[1] + w[2] + w[3]; }
	Float4 sqrt()const {
		return _mm_sqrt_ps(m128);
	}
	float operator[](int i)const { return w[i]; }
	float& operator[](int i){ return w[i]; }
	};

	class Int4{
	union {
		int w[4];
		__m128i m128;
	};
    Int4(__m128i x):m128(x){}
	public:
	    Int4(int i):m128(_mm_set1_epi32(i)){}
	    Int4():m128(_mm_setzero_si128()){}
	    Int4(int a, int b, int c, int d){
	        w[0] = a;
	        w[1] = b;
	        w[2] = c;
	        w[3] = d;
	    }
		void clear(){
			m128 = _mm_setzero_si128();
		}
	    void operator=(const Int4& i4){m128 = i4.m128;}
	    Int4 operator+(const Int4& i4)const {return _mm_add_epi32(m128, i4.m128);}
	    Int4 operator-(const Int4& i4)const {return _mm_sub_epi32(m128, i4.m128);}
	    Int4 operator*(const Int4& i4)const {return _mm_mul_epi32(m128, i4.m128);}

	    void operator+=(const Int4& i4){m128 = _mm_add_epi32(m128, i4.m128);}
	    void operator-=(const Int4& i4){m128 = _mm_sub_epi32(m128, i4.m128);}
	    void operator*=(const Int4& i4){m128 = _mm_mul_epi32(m128, i4.m128);}
		int operator[](int i)const { return w[i]; }
		int& operator[](int i){ return w[i]; }
	};
	template<size_t Size>
	class VFlt{
		alignas(32) float w[Size];
		__m128 get_m128(size_t idx)const{
			assert(idx < Size);
			assert(idx % 4 == 0);
			return _mm_load_ps(w + idx);
		}
		void set_m128(size_t idx, const __m128& m128)const{
			assert(idx < Size);
			assert(idx % 4 == 0);
			_mm_store_ps(w + idx, m128);
		}
	public:
		VFlt(){
		}
		VFlt(float f){
			__m128 mm = _mm_set1_ps(f);
			for(int i = 0;i < int64_t(Size) - 3;i+=4){
				set_m128(i, mm);
			}
			if(Size % 4 >= 3)w[Size - 3] = f;
			if(Size % 4 >= 2)w[Size - 2] = f;
			if(Size % 4 >= 1)w[Size - 1] = f;
		}
		VFlt(const VFlt<Size>& rhs){
			(*this) = rhs;
		}
		static size_t size(){return Size;}
		void clear(){
			for(int i = 0;i < int64_t(Size) - 3;i+=4){
				_mm_store_ps(w + i, _mm_setzero_ps());
			}
			if(Size % 4 >= 3)w[Size - 3] = 0;
			if(Size % 4 >= 2)w[Size - 2] = 0;
			if(Size % 4 >= 1)w[Size - 1] = 0;
		}
		void operator=(const VFlt<Size>& rhs){
			for(int i = 0;i < int64_t(Size) - 3;i+=4){
				__m128 mm = rhs.get_m128(i);
				_mm_store_ps(w + i, mm);
			}
			if(Size % 4 >= 3)w[Size - 3] = rhs.w[Size - 3];
			if(Size % 4 >= 2)w[Size - 2] = rhs.w[Size - 2];
			if(Size % 4 >= 1)w[Size - 1] = rhs.w[Size - 1];
		}
		//内積計算
		float inner_product(const VFlt<Size>& rhs)const{
			float ret = 0;
			if(Size >= 4){
				__m128 mm;
				for(int i=0;i < Size - 3;i+=4){
#ifdef FMA_ENABLE
				//FMAを用いた実装
				mm = _mm_fmadd_ps(get_m128(i), rhs.get_m128(i), mm);
#else
				mm = _mm_add_ps(mm, _mm_mul_ps(get_m128(i), rhs.get_m128(i)));
#endif
				}
				alignas(32) float v[4];
				_mm_store_ps(v, mm);
				ret = v[0] + v[1] + v[2] + v[3];
			}
			if(Size % 4 >= 3)ret = std::fma(w[Size - 3], rhs.w[Size - 3], ret);
			if(Size % 4 >= 2)ret = std::fma(w[Size - 2], rhs.w[Size - 2], ret);
			if(Size % 4 >= 1)ret = std::fma(w[Size - 1], rhs.w[Size - 1], ret);
			return ret;
		}
#define MATH_OPERATOR(OP, name)\
VFlt<Size> operator OP(const VFlt<Size>& rhs)const{\
	VFlt ret;\
	for(int i=0;i<int64_t(Size) - 3;i+=4){\
		__m128 mm = _mm_##name##_ps(_mm_load_ps(w + i), _mm_load_ps(rhs.w + i));\
		_mm_store_ps(ret.w + i, mm);\
	}\
	if(Size % 4 >= 3)ret.w[Size - 3] = w[Size - 3] OP rhs.w[Size - 3];\
	if(Size % 4 >= 2)ret.w[Size - 2] = w[Size - 2] OP rhs.w[Size - 2];\
	if(Size % 4 >= 1)ret.w[Size - 1] = w[Size - 1] OP rhs.w[Size - 1];\
}\
void operator OP##=(const VFlt<Size>& rhs){\
	for(int i=0;i<int64_t(Size) - 3;i+=4){\
		__m128 mm = _mm_##name##_ps(_mm_load_ps(w + i), _mm_load_ps(rhs.w + i));\
		_mm_store_ps(w + i, mm);\
	}\
	if(Size % 4 >= 3)w[Size - 3] OP##= rhs.w[Size - 3];\
	if(Size % 4 >= 2)w[Size - 2] OP##= rhs.w[Size - 2];\
	if(Size % 4 >= 1)w[Size - 1] OP##= rhs.w[Size - 1];\
}
		MATH_OPERATOR(+, add);
		MATH_OPERATOR(-, sub);
		MATH_OPERATOR(*, mul);
		MATH_OPERATOR(/, div);
#undef MATH_OPERATOR
		float operator[](size_t idx)const{
			assert(idx < Size);
			return w[idx];
		}
		float& operator[](size_t idx){
			assert(idx < Size);
			return w[idx];
		}
	};
}