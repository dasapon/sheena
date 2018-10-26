#pragma once

#ifdef __FMA__
#define FMA_ENABLE
#endif
#if defined(__AVX__) && defined(__AVX2__) && !defined(NO_SIMD256)
#define SIMD256_AVAILABLE
#include <immintrin.h>

#define ADD_PS _mm256_add_ps
#define SUB_PS _mm256_sub_ps
#define MUL_PS _mm256_mul_ps
#define DIV_PS _mm256_div_ps
#define FMADD_PS _mm256_fmadd_ps
#define SET1_PS _mm256_set1_ps
#define LOAD_PS _mm256_loadu_ps
#define STORE_PS _mm256_storeu_ps
#define SETZERO_PS _mm256_setzero_ps
#define FNMADD_PS _mm256_fnmadd_ps
#define SQRT_PS _mm256_sqrt_ps
#define RSQRT_PS _mm256_rsqrt_ps
#define MIN_PS _mm256_min_ps
#define MAX_PS _mm256_min_ps

#define LOAD_SI(x) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x))
#define STORE_SI(x, y) _mm256_storeu_si256(reinterpret_cast<__m256i*>(x), y)
#define SETZERO_SI _mm256_setzero_si256
#define SET1_EPI32 _mm256_set1_epi32
#define ADD_EPI32 _mm256_add_epi32
#define SUB_EPI32 _mm256_sub_epi32
#define MULLO_EPI32 _mm256_mullo_epi32
#define AND_SI _mm256_and_si256
#define OR_SI _mm256_or_si256
#define XOR_SI _mm256_xor_si256

#define CVTPS_EPI32 _mm256_cvtps_epi32
#define CVTEPI32_PS _mm256_cvtepi32_ps

#else

#define ADD_PS _mm_add_ps
#define SUB_PS _mm_sub_ps
#define MUL_PS _mm_mul_ps
#define DIV_PS _mm_div_ps
#define FMADD_PS _mm_fmadd_ps
#define SET1_PS _mm_set1_ps
#define LOAD_PS _mm_load_ps
#define STORE_PS _mm_store_ps
#define SETZERO_PS _mm_setzero_ps
#define FNMADD_PS _mm_fnmadd_ps
#define SQRT_PS _mm_sqrt_ps
#define RSQRT_PS _mm_rsqrt_ps
#define MIN_PS _mm_min_ps
#define MAX_PS _mm_max_ps

#define LOAD_SI(x) _mm_load_si128(reinterpret_cast<const __m128i*>(x))
#define STORE_SI(x, y) _mm_store_si128(reinterpret_cast<__m128i*>(x), y)
#define SETZERO_SI _mm_setzero_si128
#define SET1_EPI32 _mm_set1_epi32
#define ADD_EPI32 _mm_add_epi32
#define SUB_EPI32 _mm_sub_epi32
#define MULLO_EPI32 _mm_mullo_epi32
#define AND_SI _mm_and_si128
#define OR_SI _mm_or_si128
#define XOR_SI _mm_xor_si128

#define CVTPS_EPI32 _mm_cvtps_epi32
#define CVTEPI32_PS _mm_cvtepi32_ps

#endif
#include <nmmintrin.h>
#include <x86intrin.h>

namespace sheena{
#define MATH_OPERATOR(TYPE, VECTOR, SET1, LOAD, STORE, OP, OP_NAME)\
VECTOR operator OP(const VECTOR& rhs)const{\
	VECTOR ret;\
	size_t i = 0;\
	if(size_with_padding >= 4 * ways){\
		const size_t e = size_with_padding - size_with_padding % (ways * 4);\
		for(;i<e;i+=ways * 4){\
			STORE(ret.w + i, OP_NAME(LOAD(w + i), LOAD(rhs.w + i)));\
			STORE(ret.w + i + ways * 1, OP_NAME(LOAD(w + i + ways * 1), LOAD(rhs.w + i + ways * 1)));\
			STORE(ret.w + i + ways * 2, OP_NAME(LOAD(w + i + ways * 2), LOAD(rhs.w + i + ways * 2)));\
			STORE(ret.w + i + ways * 3, OP_NAME(LOAD(w + i + ways * 3), LOAD(rhs.w + i + ways * 3)));\
		}\
	}\
	for(;i<size_with_padding;i+=ways){\
		STORE(ret.w + i, OP_NAME(LOAD(w + i), LOAD(rhs.w + i)));\
	}\
	return ret;\
}\
void operator OP##=(const VECTOR& rhs){\
	size_t i = 0;\
	if(size_with_padding >= 4 * ways){\
		const size_t e = size_with_padding - size_with_padding % (ways * 4);\
		for(;i<e;i+=ways * 4){\
			STORE(w + i, OP_NAME(LOAD(w + i), LOAD(rhs.w + i)));\
			STORE(w + i + ways * 1, OP_NAME(LOAD(w + i + ways * 1), LOAD(rhs.w + i + ways * 1)));\
			STORE(w + i + ways * 2, OP_NAME(LOAD(w + i + ways * 2), LOAD(rhs.w + i + ways * 2)));\
			STORE(w + i + ways * 3, OP_NAME(LOAD(w + i + ways * 3), LOAD(rhs.w + i + ways * 3)));\
		}\
	}\
	for(;i<size_with_padding;i+=ways){\
		STORE(w + i, OP_NAME(LOAD(w + i), LOAD(rhs.w + i)));\
	}\
}\
VECTOR operator OP(TYPE rhs)const{\
	VECTOR ret;\
	MM rhs_mm = SET1(rhs);\
	size_t i = 0;\
	if(size_with_padding >= 4 * ways){\
		const size_t e = size_with_padding - size_with_padding % (ways * 4);\
		for(;i<e;i+=ways * 4){\
			STORE(ret.w + i, OP_NAME(LOAD(w + i), rhs_mm));\
			STORE(ret.w + i + ways * 1, OP_NAME(LOAD(w + i + ways * 1), rhs_mm));\
			STORE(ret.w + i + ways * 2, OP_NAME(LOAD(w + i + ways * 2), rhs_mm));\
			STORE(ret.w + i + ways * 3, OP_NAME(LOAD(w + i + ways * 3), rhs_mm));\
		}\
	}\
	for(;i<size_with_padding;i+=ways){\
		STORE(ret.w + i, OP_NAME(LOAD(w + i), rhs_mm));\
	}\
	return ret;\
}\
void operator OP##=(TYPE rhs){\
	size_t i = 0;\
	MM rhs_mm = SET1(rhs);\
	if(size_with_padding >= 4 * ways){\
		const size_t e = size_with_padding - size_with_padding % (ways * 4);\
		for(;i<e;i+=ways * 4){\
			STORE(w + i, OP_NAME(LOAD(w + i), rhs_mm));\
			STORE(w + i + ways * 1, OP_NAME(LOAD(w + i + ways * 1), rhs_mm));\
			STORE(w + i + ways * 2, OP_NAME(LOAD(w + i + ways * 2), rhs_mm));\
			STORE(w + i + ways * 3, OP_NAME(LOAD(w + i + ways * 3), rhs_mm));\
		}\
	}\
	for(;i<size_with_padding;i+=ways){\
		STORE(w + i, OP_NAME(LOAD(w + i), rhs_mm));\
	}\
}
	template<size_t Size>
	class VInt;
	template<size_t Size>
	class VFlt;
	template<size_t Size>
	class alignas(16) VFlt{
		friend class VInt<Size>;
#ifdef SIMD256_AVAILABLE
		static constexpr size_t ways = 8;
		using MM = __m256;
#else
		static constexpr size_t ways = 4;
		using MM = __m128;
#endif
		static MM fma(MM v1, MM v2, MM v3){
#ifdef FMA_ENABLE
			return FMADD_PS(v1, v2, v3);
#else
			return ADD_PS(MUL_PS(v1, v2), v3);
#endif
		}
		static constexpr size_t simd_loop_end = Size - Size % ways;
		static constexpr size_t padding = Size% ways != 0 ? ways - (Size % ways) : 0;
		static constexpr size_t size_with_padding = Size + padding;
		float w[Size + padding];
	public:
		VFlt(){}
		explicit VFlt(float f){
			MM mm = SET1_PS(f);
			for(size_t i = 0;i < size_with_padding;i+=ways){
				STORE_PS(w + i, mm);
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
			for(size_t i = 0;i < size_with_padding;i+=ways){
				STORE_PS(w + i, SETZERO_PS());
			}
		}
		void operator=(const VFlt<Size>& rhs){
			for(size_t i = 0;i < size_with_padding;i+=ways){
				STORE_PS(w + i, LOAD_PS(rhs.w + i));
			}
		}
		
		MATH_OPERATOR(float, VFlt<Size>, SET1_PS, LOAD_PS, STORE_PS, +, ADD_PS);
		MATH_OPERATOR(float, VFlt<Size>, SET1_PS, LOAD_PS, STORE_PS, -, SUB_PS);
		MATH_OPERATOR(float, VFlt<Size>, SET1_PS, LOAD_PS, STORE_PS, *, MUL_PS);
		MATH_OPERATOR(float, VFlt<Size>, SET1_PS, LOAD_PS, STORE_PS, /, DIV_PS);

		VFlt<Size> sqrt()const{
			VFlt<Size> ret;
			for(size_t i = 0;i < size_with_padding;i+=ways){
				STORE_PS(ret.w + i, SQRT_PS(LOAD_PS(w + i)));
			}
			return ret;
		}
		VFlt<Size> rsqrt()const{
			VFlt<Size> ret;
			for(size_t i = 0;i < size_with_padding;i+=ways){
				STORE_PS(ret.w + i, RSQRT_PS(LOAD_PS(w + i)));
			}
			return ret;
		}
		VFlt<Size> exp()const{
			VFlt<Size> ret;
			for(int i=0;i<Size;i++){
				ret[i] = std::exp(w[i]);
			}
			return ret;
		}
		float max()const{
			float ret = -FLT_MAX;
			size_t i = 0;
			if(Size >= ways){
				MM mm = SET1_PS(ret);
				if(Size >= ways * 4){
					MM mm2 = mm, mm3 = mm, mm4 = mm;
					const size_t e = Size - Size % (ways * 4);
					for(;i < e;i+=ways * 4){
						mm = MAX_PS(LOAD_PS(w + i), mm);
						mm2 = MAX_PS(LOAD_PS(w + i + ways * 1), mm2);
						mm3 = MAX_PS(LOAD_PS(w + i + ways * 2), mm3);
						mm4 = MAX_PS(LOAD_PS(w + i + ways * 3), mm4);
					}
					mm = MAX_PS(mm, mm2);
					mm3 = MAX_PS(mm3, mm4);
					mm = MAX_PS(mm, mm3);
				}
				for(;i < simd_loop_end;i+=ways){
					mm = MAX_PS(LOAD_PS(w + i), mm);
				}
				alignas(ways * 4) float v[ways];
				STORE_PS(v, mm);
#ifdef SIMD256_AVAILABLE
				ret = std::max(std::min(std::min(v[0] + v[1]), std::max(v[2], v[3])), std::max(std::min(v[4], v[5]) + std::max(v[6], v[7])));
#else
				ret = std::max(std::max(v[0] + v[1]), std::max(v[2], v[3]));
#endif
			}
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end;i<Size;i++){
					ret = std::max(ret, w[i]);
				}
			}
			return ret;
		}
		float min()const{
			float ret = FLT_MAX;
			size_t i = 0;
			if(Size >= ways){
				MM mm = SET1_PS(ret);
				if(Size >= ways * 4){
					MM mm2 = mm, mm3 = mm, mm4 = mm;
					const size_t e = Size - Size % (ways * 4);
					for(;i < e;i+=ways * 4){
						mm = MIN_PS(LOAD_PS(w + i), mm);
						mm2 = MIN_PS(LOAD_PS(w + i + ways * 1), mm2);
						mm3 = MIN_PS(LOAD_PS(w + i + ways * 2), mm3);
						mm4 = MIN_PS(LOAD_PS(w + i + ways * 3), mm4);
					}
					mm = MIN_PS(mm, mm2);
					mm3 = MIN_PS(mm3, mm4);
					mm = MIN_PS(mm, mm3);
				}
				for(;i < simd_loop_end;i+=ways){
					mm = MIN_PS(LOAD_PS(w + i), mm);
				}
				alignas(ways * 4) float v[ways];
				STORE_PS(v, mm);
#ifdef SIMD256_AVAILABLE
				ret = std::min(std::min(std::min(v[0] + v[1]), std::min(v[2], v[3])), std::min(std::min(v[4], v[5]) + std::min(v[6], v[7])));
#else
				ret = std::min(std::min(v[0] + v[1]), std::min(v[2], v[3]));
#endif
			}
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end;i<Size;i++){
					ret = std::min(ret, w[i]);
				}
			}
			return ret;
		}
		//合計値の計算
		float sum()const{
			float ret = 0;
			size_t i = 0;
			if(Size >= ways){
				MM mm = SETZERO_PS();
				if(Size >= ways * 4){
					MM mm2 = SETZERO_PS(), mm3 = SETZERO_PS(), mm4 = SETZERO_PS();
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
				alignas(ways * 4) float v[ways];
				STORE_PS(v, mm);
#ifdef SIMD256_AVAILABLE
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
#else
				ret = (v[0] + v[1]) + (v[2] + v[3]);
#endif
			}
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
				MM mm = SETZERO_PS();
				//インライン展開
				size_t i = 0;
				if(Size >= ways * 4){
					MM mm2 = SETZERO_PS(), mm3 = SETZERO_PS(), mm4 = SETZERO_PS();
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
				alignas(ways * 4) float v[ways];
				STORE_PS(v, mm);
#ifdef SIMD256_AVAILABLE
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
#else
				ret = (v[0] + v[1]) + (v[2] + v[3]);
#endif
			}
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end;i<Size;i++){
					ret = std::fma(w[i], rhs.w[i], ret);
				}
			}
			return ret;
		}

		void add_product(const VFlt<Size>& v1, const VFlt<Size>& v2){
			size_t i = 0;
			if(size_with_padding >= 4 * ways){\
				const size_t e = size_with_padding - size_with_padding % (ways * 4);\
				for(;i<e;i+=ways * 4){
					STORE_PS(w + i, fma(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
					STORE_PS(w + i + ways * 1, fma(LOAD_PS(v1.w + i + ways * 1), LOAD_PS(v2.w + i + ways * 1), LOAD_PS(w + i + ways * 1)));
					STORE_PS(w + i + ways * 2, fma(LOAD_PS(v1.w + i + ways * 2), LOAD_PS(v2.w + i + ways * 2), LOAD_PS(w + i + ways * 2)));
					STORE_PS(w + i + ways * 3, fma(LOAD_PS(v1.w + i + ways * 3), LOAD_PS(v2.w + i + ways * 3), LOAD_PS(w + i + ways * 3)));
				}
			}
			for(;i<size_with_padding;i+=ways){
				STORE_PS(w + i, fma(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
			}
		}
		void sub_product(const VFlt<Size>& v1, const VFlt<Size>& v2){
#ifdef FMA_ENABLE
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_PS(w + i, FNMADD_PS(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
			}
#else
			operator-=(v1 * v2);
#endif
		}
		VInt<Size> to_vint()const;
	};
	template<size_t Size>
	class alignas(16) VInt{
		friend class VFlt<Size>;
#ifdef SIMD256_AVAILABLE
		static constexpr size_t simd_loop_end = Size - Size % 8;
		using MM = __m256i;
		static constexpr size_t ways = 8;
#else
		static constexpr size_t simd_loop_end = Size - Size % 4;
		using MM = __m128i;
		static constexpr size_t ways = 4;
#endif
		static constexpr size_t padding = Size% ways != 0 ? ways - (Size % ways) : 0;
		static constexpr size_t size_with_padding = Size + padding;
		int32_t w[size_with_padding];
	public:
		VInt(){}
		explicit VInt(int x){
			MM x_mm = SET1_EPI32(x);
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, x_mm);
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
		static constexpr size_t size(){return Size;}
		void clear(){
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, SETZERO_SI());
			}
		}
		void operator=(const VInt& rhs){
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, LOAD_SI(rhs. w + i));
			}
		}
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, +, ADD_EPI32);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, -, SUB_EPI32);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, *, MULLO_EPI32);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, &, AND_SI);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, |, OR_SI);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, ^, XOR_SI);
		VFlt<Size> to_vflt()const;
	};
	template<size_t Size>
	VInt<Size> VFlt<Size>::to_vint()const{
		VInt<Size> ret;
		for(size_t i=0;i<size_with_padding;i+=ways){
			STORE_SI(ret.w + i, CVTPS_EPI32(LOAD_PS(w + i)));
		}
		return ret;
	}
	template<size_t Size>
	VFlt<Size> VInt<Size>::to_vflt()const{
		VFlt<Size> ret;
		for(size_t i=0;i<size_with_padding;i+=ways){
			STORE_PS(ret.w + i, CVTEPI32_PS(LOAD_SI(w + i)));
		}
		return ret;
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
#undef FNMADD_PS

#undef LOAD_SI
#undef STORE_SI 
#undef SETZERO_SI
#undef SET1_EPI
#undef ADD_EPI
#undef SUB_EPI
#undef MULLO_EPI
#undef AND_SI
#undef OR_SI
#undef XOR_SI
#undef MATH_OPERATOR

	class Float4 : VFlt<4>{
	public:
		Float4(float f):VFlt<4>(f){}
		Float4(){}
		Float4(const Float4& f4){
			operator=(f4);
		}
		Float4(float w0, float w1,float w2, float w3){
			operator[](0) = w0;
			operator[](1) = w1;
			operator[](2) = w2;
			operator[](3) = w3;
		}
		using VFlt<4>::clear;
		using VFlt<4>::operator=;
		using VFlt<4>::operator+;
		using VFlt<4>::operator-;
		using VFlt<4>::operator*;
		using VFlt<4>::operator/;
		using VFlt<4>::operator+=;
		using VFlt<4>::operator-=;
		using VFlt<4>::operator*=;
		using VFlt<4>::operator/=;
		using VFlt<4>::sum;
		using VFlt<4>::sqrt;
		using VFlt<4>::operator[];
	};

	class Int4 : VInt<4>{
	public:
	    Int4(int i):VInt<4>(i){}
	    Int4(){
			clear();
		}
	    Int4(int w0, int w1, int w2, int w3){
			operator[](0) = w0;
			operator[](1) = w1;
			operator[](2) = w2;
			operator[](3) = w3;
	    }
		using VInt<4>::clear;
		using VInt<4>::operator=;
		using VInt<4>::operator+;
		using VInt<4>::operator-;
		using VInt<4>::operator*;
		using VInt<4>::operator+=;
		using VInt<4>::operator-=;
		using VInt<4>::operator*=;
		using VInt<4>::operator[];
	};

}