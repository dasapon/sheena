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
	class alignas(16) VFlt{
#ifdef SIMD256_ENABLE
		static constexpr size_t ways = 8;
		using __MSIMD = __m256;
#define ADD_PS _mm256_add_ps
#define SUB_PS _mm256_sub_ps
#define MUL_PS _mm256_mul_ps
#define DIV_PS _mm256_div_ps
#define FMADD_PS _mm256_fmadd_ps
#define SET1_PS _mm256_set1_ps
#define LOAD_PS _mm256_loadu_ps
#define STORE_PS _mm256_storeu_ps
#define SETZERO_PS _mm256_setzero_ps
#else
		static constexpr size_t ways = 4;
		using __MSIMD = __m128;
#define ADD_PS _mm_add_ps
#define SUB_PS _mm_sub_ps
#define MUL_PS _mm_mul_ps
#define DIV_PS _mm_div_ps
#define FMADD_PS _mm_fmadd_ps
#define SET1_PS _mm_set1_ps
#define LOAD_PS _mm_load_ps
#define STORE_PS _mm_store_ps
#define SETZERO_PS _mm_setzero_ps
#endif
		static __MSIMD fma(__MSIMD v1, __MSIMD v2, __MSIMD v3){
#ifdef FMA_ENABLE
			return FMADD_PS(v1, v2, v3);
#else
			return ADD_PS(MUL_PS(v1, v2), v3);
#endif
		}
		static constexpr size_t simd_loop_end = Size - Size % ways;
		float w[Size];
	public:
		VFlt(){}
		VFlt(float f){
			__MSIMD mm = SET1_PS(f);
			for(size_t i = 0;i < simd_loop_end;i+=ways){
				STORE_PS(w + i, mm);
			}
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
			for(size_t i = 0;i < simd_loop_end;i+=ways){
				STORE_PS(w + i, SETZERO_PS());
			}
			if(Size != simd_loop_end){
				for(size_t i = simd_loop_end; i<Size; i++){
					w[i] = 0;
				}
			}
		}
		void operator=(const VFlt<Size>& rhs){
			for(size_t i = 0;i < simd_loop_end;i+=ways){
				STORE_PS(w + i, LOAD_PS(rhs.w + i));
			}
			if(Size != simd_loop_end){
				for(size_t i = simd_loop_end; i<Size; i++){
					w[i] = rhs.w[i];
				}
			}
		}
		//合計値の計算
		float sum()const{
			float ret = 0;
			size_t i = 0;
			if(Size >= ways){
				__MSIMD mm = SETZERO_PS();
				if(Size >= ways * 4){
					__MSIMD mm2 = SETZERO_PS(), mm3 = SETZERO_PS(), mm4 = SETZERO_PS();
					const size_t e = Size - Size % (ways * 4);
					for(;i < e;i+=ways * 4){
						mm = ADD_PS(LOAD_PS(w + i), mm);
						mm2 = ADD_PS(LOAD_PS(w + i + ways * 1), mm2);
						mm3 = ADD_PS(LOAD_PS(w + i + ways * 2), mm3);
						mm4 = ADD_PS(LOAD_PS(w + i + ways * 3), mm4);
					}
					mm = ADD_PS(mm, mm2);
					mm3 = ADD_PS(mm3, mm4);
					mm = ADD_PS(mm, mm3);
				}
				for(;i < simd_loop_end;i+=ways){
					mm = ADD_PS(LOAD_PS(w + i), mm);
				}
#ifdef SIMD256_ENABLE
				alignas(32) float v[8];
				_mm256_store_ps(v, mm);
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
			}
#else
				alignas(16) float v[4];
				_mm_store_ps(v, mm);
				ret = (v[0] + v[1]) + (v[2] + v[3]);
			}
#endif
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end;i<Size;i++){
					ret += w[i];
				}
			}
			return ret;
		}
		//内積計算
		float inner_product(const VFlt<Size>& rhs)const{
			float ret = 0;
			if(Size >= ways){
				__MSIMD mm = SETZERO_PS();
				//インライン展開
				size_t i = 0;
				if(Size >= ways * 4){
					__MSIMD mm2 = SETZERO_PS(), mm3 = SETZERO_PS(), mm4 = SETZERO_PS();
					const size_t e = Size - Size % (ways * 4);
					for(;i < e;i+=ways * 4){
						mm = fma(LOAD_PS(w + i), LOAD_PS(rhs.w + i), mm);
						mm2 = fma(LOAD_PS(w + i + ways * 1), LOAD_PS(rhs.w + i + ways * 1), mm2);
						mm3 = fma(LOAD_PS(w + i + ways * 2), LOAD_PS(rhs.w + i + ways * 2), mm3);
						mm4 = fma(LOAD_PS(w + i + ways * 3), LOAD_PS(rhs.w + i + ways * 3), mm4);
					}
					mm = ADD_PS(mm, mm2);
					mm3 = ADD_PS(mm3, mm4);
					mm = ADD_PS(mm, mm3);
				}
				for(;i < simd_loop_end;i+=ways){
					mm = fma(LOAD_PS(w + i), LOAD_PS(rhs.w + i), mm);
				}
#ifdef SIMD256_ENABLE
				alignas(32) float v[8];
				_mm256_store_ps(v, mm);
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
			}
#else
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
#define MATH_OPERATOR(OP, OP_NAME)\
VFlt<Size> operator OP(const VFlt<Size>& rhs)const{\
	VFlt ret;\
	size_t i = 0;\
	if(Size >= 4 * ways){\
		const size_t e = Size - Size % (ways * 4);\
		for(;i<e;i+=ways * 4){\
			STORE_PS(ret.w + i, OP_NAME(LOAD_PS(w + i), LOAD_PS(rhs.w + i)));\
			STORE_PS(ret.w + i + ways * 1, OP_NAME(LOAD_PS(w + i + ways * 1), LOAD_PS(rhs.w + i + ways * 1)));\
			STORE_PS(ret.w + i + ways * 2, OP_NAME(LOAD_PS(w + i + ways * 2), LOAD_PS(rhs.w + i + ways * 2)));\
			STORE_PS(ret.w + i + ways * 3, OP_NAME(LOAD_PS(w + i + ways * 3), LOAD_PS(rhs.w + i + ways * 3)));\
		}\
	}\
	for(;i<simd_loop_end;i+=ways){\
		STORE_PS(ret.w + i, OP_NAME(LOAD_PS(w + i), LOAD_PS(rhs.w + i)));\
	}\
	if(Size != simd_loop_end){\
		for(;i<Size;i++){\
			ret.w[i] = w[i] OP rhs.w[i];\
		}\
	}\
	return ret;\
}\
void operator OP##=(const VFlt<Size>& rhs){\
	size_t i = 0;\
	if(Size >= 4 * ways){\
		const size_t e = Size - Size % (ways * 4);\
		for(;i<e;i+=ways * 4){\
			STORE_PS(w + i, OP_NAME(LOAD_PS(w + i), LOAD_PS(rhs.w + i)));\
			STORE_PS(w + i + ways * 1, OP_NAME(LOAD_PS(w + i + ways * 1), LOAD_PS(rhs.w + i + ways * 1)));\
			STORE_PS(w + i + ways * 2, OP_NAME(LOAD_PS(w + i + ways * 2), LOAD_PS(rhs.w + i + ways * 2)));\
			STORE_PS(w + i + ways * 3, OP_NAME(LOAD_PS(w + i + ways * 3), LOAD_PS(rhs.w + i + ways * 3)));\
		}\
	}\
	for(;i<simd_loop_end;i+=ways){\
		STORE_PS(w + i, OP_NAME(LOAD_PS(w + i), LOAD_PS(rhs.w + i)));\
	}\
	if(Size != simd_loop_end){\
		for(;i<Size;i++){\
			w[i] OP##= rhs.w[i];\
		}\
	}\
}
		MATH_OPERATOR(+, ADD_PS);
		MATH_OPERATOR(-, SUB_PS);
		MATH_OPERATOR(*, MUL_PS);
		MATH_OPERATOR(/, DIV_PS);
#undef MATH_OPERATOR
		void fmadd(const VFlt<Size>& v1, const VFlt<Size>& v2){
			size_t i = 0;
			if(Size > 4 * ways){
				const size_t e = Size - Size % (ways * 4);
				for(;i<e;i+=ways * 4){
					STORE_PS(w + i, fma(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
					STORE_PS(w + i + ways * 1, fma(LOAD_PS(v1.w + i + ways * 1), LOAD_PS(v2.w + i + ways * 1), LOAD_PS(w + i + ways * 1)));
					STORE_PS(w + i + ways * 2, fma(LOAD_PS(v1.w + i + ways * 2), LOAD_PS(v2.w + i + ways * 2), LOAD_PS(w + i + ways * 2)));
					STORE_PS(w + i + ways * 3, fma(LOAD_PS(v1.w + i + ways * 3), LOAD_PS(v2.w + i + ways * 3), LOAD_PS(w + i + ways * 3)));
				}
			}
			for(;i<simd_loop_end;i+=ways){
				STORE_PS(w + i, fma(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
			}
			if(Size != simd_loop_end){
				for(;i<Size;i++){
					w[i] = std::fma(v1.w[i], v2.w[i], w[i]);
				}
			}
		}
		void fmsub(const VFlt<Size>& v1, const VFlt<Size>& v2){
#ifdef FMA_ENABLE
			for(size_t i=0;i<simd_loop_end;i+=ways){
#ifdef SIMD256_ENABLE
				STORE_PS(w + i, _mm256_fnmadd_ps(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
#else
				STORE_PS(w + i, _mm_fnmadd_ps(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
#endif
			}
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end;i<Size;i++){
					w[i] = std::fma(-v1.w[i], v2.w[i], w[i]);
				}
			}
#else
			operator-=(v1 * v2);
#endif
		}
#undef ADD_PS
#undef SUB_PS
#undef MUL_PS
#undef DIV_PS
#undef FMADD_PS 
#undef SET1_PS
#undef LOAD_PS
#undef STORE_PS
#undef SETZERO_PS
	};
	template<size_t Size>
	class alignas(16) VInt{
#ifdef SIMD256_ENABLE
		static constexpr size_t simd_loop_end = Size - Size % 8;
		using __MSIMD = __m256i;
		__MSIMD load(size_t idx)const{
			return _mm256_loadu_si256(reinterpret_cast<const __MSIMD*>(w + idx));
		}
		static constexpr size_t inc = 8;
#else
		static constexpr size_t simd_loop_end = Size - Size % 4;
		using __MSIMD = __m128i;
		__MSIMD load(size_t idx)const{
			return _mm_load_si128(reinterpret_cast<const __MSIMD*>(w + idx));
		}
		static constexpr size_t inc = 4;
#endif
		int32_t w[Size];
	public:
		VInt(){}
		VInt(int x){
			for(size_t i=0;i<simd_loop_end;i+=inc){
#ifdef SIMD256_ENABLE
				_mm256_storeu_si256(reinterpret_cast<__MSIMD*>(w + i), _mm256_set1_epi32(x));
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
				_mm256_storeu_si256(reinterpret_cast<__MSIMD*>(w + i), _mm256_setzero_si256());
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
				_mm256_storeu_si256(reinterpret_cast<__MSIMD*>(w + i), rhs.load(i));
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
		MATH_OPERATOR(_mm256_storeu_si256, +, _mm256_add_epi32);
		MATH_OPERATOR(_mm256_storeu_si256, -, _mm256_sub_epi32);
		MATH_OPERATOR(_mm256_storeu_si256, *, _mm256_mullo_epi32);
		MATH_OPERATOR(_mm256_storeu_si256, &, _mm256_and_si256);
		MATH_OPERATOR(_mm256_storeu_si256, |, _mm256_or_si256);
		MATH_OPERATOR(_mm256_storeu_si256, ^, _mm256_xor_si256);
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