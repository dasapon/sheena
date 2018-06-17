#pragma once

#ifdef __FMA__
#define FMA_ENABLE
#endif
#if defined(__AVX__) && defined(__AVX2__) && !defined(NO_SIMD256)
#define SIMD256_ENABLE
#include <immintrin.h>
#endif
#include <nmmintrin.h>
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
	    Int4 operator*(const Int4& i4)const {return _mm_mullo_epi32(m128, i4.m128);}

	    void operator+=(const Int4& i4){m128 = _mm_add_epi32(m128, i4.m128);}
	    void operator-=(const Int4& i4){m128 = _mm_sub_epi32(m128, i4.m128);}
	    void operator*=(const Int4& i4){m128 = _mm_mullo_epi32(m128, i4.m128);}
		int operator[](int i)const { return w[i]; }
		int& operator[](int i){ return w[i]; }
	};

	template<size_t Size>
#ifdef SIMD256_ENABLE
	class alignas(32) VFlt{
		static constexpr size_t simd_loop_end = Size - Size % 8;
		using __MSIMD = __m256;
#else
	class alignas(16) VFlt{
		static constexpr size_t simd_loop_end = Size - Size % 4;
		using __MSIMD = __m128;
#endif
		float w[Size];
	public:
		VFlt(){}
		VFlt(float f){
#ifdef SIMD256_ENABLE
			__MSIMD mm = _mm256_set1_ps(f);
			for(size_t i = 0;i < simd_loop_end;i+=8){
				_mm256_store_ps(w + i, mm);
			}
#else
			__MSIMD mm = _mm_set1_ps(f);
			for(size_t i = 0;i < simd_loop_end;i+=4){
				_mm_store_ps(w + i, mm);
			}
#endif
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end; i < Size;i++)w[i] = f;
			}
		}
		VFlt(const VFlt<Size>& rhs){
			(*this) = rhs;
		}
		float operator[](size_t idx)const{
			assert(idx < Size);
			return w[idx];
		}
		float& operator[](size_t idx){
			assert(idx < Size);
			return w[idx];
		}
		static constexpr size_t size(){return Size;}
		void clear(){
#ifdef SIMD256_ENABLE
			for(size_t i = 0;i < simd_loop_end;i+=8){
				_mm256_store_ps(w + i, _mm256_setzero_ps());
			}
#else
			for(size_t i = 0;i < simd_loop_end;i+=4){
				_mm_store_ps(w + i, _mm_setzero_ps());
			}
#endif
			if(Size != simd_loop_end){
				for(size_t i = simd_loop_end; i<Size; i++){
					w[i] = 0;
				}
			}
		}
		void operator=(const VFlt<Size>& rhs){
#ifdef SIMD256_ENABLE
			for(size_t i = 0;i < simd_loop_end;i+=8){
				_mm256_store_ps(w + i, _mm256_load_ps(rhs.w + i));
			}
#else
			for(size_t i = 0;i < simd_loop_end;i+=4){
				_mm_store_ps(w + i, _mm_load_ps(rhs.w + i));
			}
#endif
			if(Size != simd_loop_end){
				for(size_t i = simd_loop_end; i<Size; i++){
					w[i] = rhs.w[i];
				}
			}
		}
		//内積計算
		float inner_product(const VFlt<Size>& rhs)const{
			float ret = 0;
#ifdef SIMD256_ENABLE
			if(Size >= 8){
				__MSIMD mm = _mm256_setzero_ps();
				//インライン展開
				size_t i = 0;
				if(Size >= 32){
					__MSIMD mm2, mm3, mm4;
					const size_t e = Size - Size % 32;
					for(;i < e;i+=32){
#ifdef FMA_ENABLE
						//FMAを用いた実装
						mm = _mm256_fmadd_ps(_mm256_load_ps(w + i), _mm256_load_ps(rhs.w + i), mm);
						mm2 = _mm256_fmadd_ps(_mm256_load_ps(w + i + 8), _mm256_load_ps(rhs.w + i + 8), mm2);
						mm3 = _mm256_fmadd_ps(_mm256_load_ps(w + i + 16), _mm256_load_ps(rhs.w + i + 18), mm3);
						mm4 = _mm256_fmadd_ps(_mm256_load_ps(w + i + 24), _mm256_load_ps(rhs.w + i + 24), mm4);
#else
						mm = _mm256_add_ps(mm, _mm256_mul_ps(_mm256_load_ps(w + i), _mm256_load_ps(rhs.w + i)));
						mm2 = _mm256_add_ps(mm2, _m256_mul_ps(_mm256_load_ps(w + i + 8), _mm256_load_ps(rhs.w + i + 8)));
						mm3 = _mm256_add_ps(mm3, _mm256_mul_ps(_mm256_load_ps(w + i + 16), _mm256_load_ps(rhs.w + i + 16)));
						mm4 = _mm256_add_ps(mm4, _mm256_mul_ps(_mm256_load_ps(w + i + 24), _mm256_load_ps(rhs.w + i + 24)));
#endif
					}
					mm = _mm256_add_ps(mm, mm2);
					mm3 = _mm256_add_ps(mm3, mm4);
					mm = _mm256_add_ps(mm, mm3);
				}
				for(;i < simd_loop_end;i+=8){
#ifdef FMA_ENABLE
				//FMAを用いた実装
				mm = _mm256_fmadd_ps(_mm256_load_ps(w + i), _mm256_load_ps(rhs.w + i), mm);
#else
				mm = _mm256_add_ps(mm, _mm256_mul_ps(_mm256_load_ps(w + i), _mm256_load_ps(rhs.w + i)));
#endif
				}
				alignas(32) float v[8];
				_mm256_store_ps(v, mm);
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
			}
#else
			if(Size >= 4){
				__MSIMD mm = _mm_setzero_ps();
				//インライン展開
				size_t i = 0;
				if(Size >= 16){
					__MSIMD mm2, mm3, mm4;
					const size_t e = Size - Size % 16;
					for(;i < e;i+=16){
#ifdef FMA_ENABLE
						//FMAを用いた実装
						mm = _mm_fmadd_ps(_mm_load_ps(w + i), _mm_load_ps(rhs.w + i), mm);
						mm2 = _mm_fmadd_ps(_mm_load_ps(w + i + 4), _mm_load_ps(rhs.w + i + 4), mm2);
						mm3 = _mm_fmadd_ps(_mm_load_ps(w + i + 8), _mm_load_ps(rhs.w + i + 8), mm3);
						mm4 = _mm_fmadd_ps(_mm_load_ps(w + i + 12), _mm_load_ps(rhs.w + i + 12), mm4);
#else
						mm = _mm_add_ps(mm, _mm_mul_ps(_mm_load_ps(w + i), _mm_load_ps(rhs.w + i)));
						mm2 = _mm_add_ps(mm2, _mm_mul_ps(_mm_load_ps(w + i + 4), _mm_load_ps(rhs.w + i + 4)));
						mm3 = _mm_add_ps(mm3, _mm_mul_ps(_mm_load_ps(w + i + 8), _mm_load_ps(rhs.w + i + 8)));
						mm4 = _mm_add_ps(mm4, _mm_mul_ps(_mm_load_ps(w + i + 12), _mm_load_ps(rhs.w + i + 12)));
#endif
					}
					mm = _mm_add_ps(mm, mm2);
					mm3 = _mm_add_ps(mm3, mm4);
					mm = _mm_add_ps(mm, mm3);
				}
				for(;i < simd_loop_end;i+=4){
#ifdef FMA_ENABLE
				//FMAを用いた実装
				mm = _mm_fmadd_ps(_mm_load_ps(w + i), _mm_load_ps(rhs.w + i), mm);
#else
				mm = _mm_add_ps(mm, _mm_mul_ps(_mm_load_ps(w + i), _mm_load_ps(rhs.w + i)));
#endif
				}
				alignas(16) float v[4];
				_mm_store_ps(v, mm);
				ret = (v[0] + v[1]) + (v[2] + v[3]);
			}
#endif
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end;i<Size;i++){
					ret = std::fma(w[i], rhs.w[i], ret);
				}
			}
			return ret;
		}
#define MATH_OPERATOR(LOAD, STORE, OP, OP_NAME, INC)\
VFlt<Size> operator OP(const VFlt<Size>& rhs)const{\
	VFlt ret;\
	for(size_t i=0;i<simd_loop_end;i+=INC){\
		__MSIMD mm = OP_NAME(LOAD(w + i), LOAD(rhs.w + i));\
		STORE(ret.w + i, mm);\
	}\
	if(Size != simd_loop_end){\
		for(size_t i=simd_loop_end;i<Size;i++){\
			ret.w[i] = w[i] OP rhs.w[i];\
		}\
	}\
	return ret;\
}\
void operator OP##=(const VFlt<Size>& rhs){\
	for(size_t i=0;i<simd_loop_end;i+=INC){\
		__MSIMD mm = OP_NAME(LOAD(w + i), LOAD(rhs.w + i));\
		STORE(w + i, mm);\
	}\
	if(Size != simd_loop_end){\
		for(size_t i=simd_loop_end;i<Size;i++){\
			w[i] OP##= rhs.w[i];\
		}\
	}\
}
#ifdef SIMD256_ENABLE
		MATH_OPERATOR(_mm256_load_ps, _mm256_store_ps, +, _mm256_add_ps, 8);
		MATH_OPERATOR(_mm256_load_ps, _mm256_store_ps, -, _mm256_sub_ps, 8);
		MATH_OPERATOR(_mm256_load_ps, _mm256_store_ps, *, _mm256_mul_ps, 8);
		MATH_OPERATOR(_mm256_load_ps, _mm256_store_ps, /, _mm256_div_ps, 8);
#else
		MATH_OPERATOR(_mm_load_ps, _mm_store_ps, +, _mm_add_ps, 4);
		MATH_OPERATOR(_mm_load_ps, _mm_store_ps, -, _mm_sub_ps, 4);
		MATH_OPERATOR(_mm_load_ps, _mm_store_ps, *, _mm_mul_ps, 4);
		MATH_OPERATOR(_mm_load_ps, _mm_store_ps, /, _mm_div_ps, 4);
#endif
#undef MATH_OPERATOR
	};
	template<size_t Size>
#ifdef SIMD256_ENABLE
	class alignas(32) VInt{
		static constexpr size_t simd_loop_end = Size - Size % 8;
		using __MSIMD = __m256i;
		__MSIMD load(size_t idx)const{
			return _mm256_load_si256(reinterpret_cast<const __MSIMD*>(w + idx));
		}
		static constexpr size_t inc = 8;
#else
	class VInt{
		static constexpr size_t simd_loop_end = Size - Size % 4;
		using __MSIMD = __m128i;
		__MSIMD load(size_t idx)const{
			return _mm_load_si128(reinterpret_cast<const __MSIMD*>(w + idx));
		}
		static constexpr size_t inc = 4;
#endif
		alignas(32) int32_t w[Size];
	public:
		VInt(){}
		VInt(int x){
			for(size_t i=0;i<simd_loop_end;i+=inc){
#ifdef SIMD256_ENABLE
				_mm256_store_si256(reinterpret_cast<__MSIMD*>(w + i), _mm256_set1_epi32(x));
#else
				_mm_store_si128(reinterpret_cast<__MSIMD*>(w + i), _mm_set1_epi32(x));
#endif
			}
			if(Size != simd_loop_end){
				for(int i=simd_loop_end;i<Size;i++)w[i] = x;
			}
		}
		VInt(const VInt<Size>& rhs){
			(*this) = rhs;
		}
		int operator[](size_t idx)const{
			assert(idx < Size);
			return w[idx];
		}
		int& operator[](size_t idx){
			assert(idx < Size);
			return w[idx];
		}
		static size_t size(){return Size;}
		void clear(){
			for(size_t i=0;i<simd_loop_end;i+=inc){
#ifdef SIMD256_ENABLE
				_mm256_store_si256(reinterpret_cast<__MSIMD*>(w + i), _mm256_setzero_si256());
#else
				_mm_store_si128(reinterpret_cast<__MSIMD*>(w + i), _mm_setzero_si128());
#endif
			}
			if(Size != simd_loop_end){
				for(int i=simd_loop_end;i<Size;i++)w[i] = 0;
			}
		}
		void operator=(const VInt& rhs){
			for(size_t i=0;i<simd_loop_end;i+=inc){
#ifdef SIMD256_ENABLE
				_mm256_store_si256(reinterpret_cast<__MSIMD*>(w + i), rhs.load(i));
#else
				_mm_store_si128(reinterpret_cast<__MSIMD*>(w + i), rhs.load(i));
#endif
			}
			if(Size != simd_loop_end){
				for(int i=simd_loop_end;i<Size;i++)w[i] = rhs.w[i];
			}
		}
#define MATH_OPERATOR(STORE, OP, OP_NAME)\
VInt<Size> operator OP(const VInt<Size>& rhs)const{\
	VInt ret;\
	for(size_t i=0;i<simd_loop_end;i+=inc){\
		__MSIMD mm = OP_NAME(load(i), rhs.load(i));\
		STORE(reinterpret_cast<__MSIMD*>(ret.w + i), mm);\
	}\
	if(Size != simd_loop_end){\
		for(size_t i=simd_loop_end;i<Size;i++){\
			ret.w[i] = w[i] OP rhs.w[i];\
		}\
	}\
	return ret;\
}\
void operator OP##=(const VInt<Size>& rhs){\
	for(size_t i=0;i<simd_loop_end;i+=inc){\
		__MSIMD mm = OP_NAME(load(i), rhs.load(i));\
		STORE(reinterpret_cast<__MSIMD*>(w + i), mm);\
	}\
	if(Size != simd_loop_end){\
		for(size_t i=simd_loop_end;i<Size;i++){\
			w[i] OP##= rhs.w[i];\
		}\
	}\
}
#ifdef SIMD256_ENABLE
		MATH_OPERATOR(_mm256_store_si256, +, _mm256_add_epi32);
		MATH_OPERATOR(_mm256_store_si256, -, _mm256_sub_epi32);
		MATH_OPERATOR(_mm256_store_si256, *, _mm256_mullo_epi32);
		MATH_OPERATOR(_mm256_store_si256, &, _mm256_and_si256);
		MATH_OPERATOR(_mm256_store_si256, |, _mm256_or_si256);
		MATH_OPERATOR(_mm256_store_si256, ^, _mm256_xor_si256);
#else
		MATH_OPERATOR(_mm_store_si128, +, _mm_add_epi32);
		MATH_OPERATOR(_mm_store_si128, -, _mm_sub_epi32);
		MATH_OPERATOR(_mm_store_si128, *, _mm_mullo_epi32);
		MATH_OPERATOR(_mm_store_si128, &, _mm_and_si128);
		MATH_OPERATOR(_mm_store_si128, |, _mm_or_si128);
		MATH_OPERATOR(_mm_store_si128, ^, _mm_xor_si128);
#endif
#undef MATH_OPERATOR
	};
}