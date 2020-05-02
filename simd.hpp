#pragma once

#ifdef __FMA__
#define FMA_ENABLE
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else 
#include <x86intrin.h>
#endif

#if defined(__AVX512F__) && !defined(NO_SIMD256)
#define SIMD512_AVAILABLE

#define ADD_PS _mm512_add_ps
#define SUB_PS _mm512_sub_ps
#define MUL_PS _mm512_mul_ps
#define DIV_PS _mm512_div_ps
#define FMADD_PS _mm512_fmadd_ps
#define SET1_PS _mm512_set1_ps
#define LOAD_PS _mm512_loadu_ps
#define STORE_PS _mm512_storeu_ps
#define SETZERO_PS _mm512_setzero_ps
#define FNMADD_PS _mm512_fnmadd_ps
#define SQRT_PS _mm512_sqrt_ps
#define RSQRT_PS _mm512_rsqrt14_ps
#define MIN_PS _mm512_min_ps
#define MAX_PS _mm512_max_ps

#define LOAD_SI(x) _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x))
#define STORE_SI(x, y) _mm512_storeu_si512(reinterpret_cast<__m512i*>(x), y)
#define SETZERO_SI _mm512_setzero_si512
#define SET1_EPI8 _mm512_set1_epi8
#define SET1_EPI16 _mm512_set1_epi16
#define SET1_EPI32 _mm512_set1_epi32
#define ADD_EPI8 _mm512_add_epi8
#define ADD_EPI16 _mm512_add_epi16
#define ADD_EPI32 _mm512_add_epi32
#define SUB_EPI8 _mm512_sub_epi8
#define SUB_EPI16 _mm512_sub_epi16
#define SUB_EPI32 _mm512_sub_epi32
#define MULLO_EPI16 _mm512_mullo_epi16
#define MULLO_EPI32 _mm512_mullo_epi32
#define AND_SI _mm512_and_si512
#define ANDNOT_SI _mm512_andnot_si512
#define OR_SI _mm512_or_si512
#define XOR_SI _mm512_xor_si512
#define SLLI_EPI16 _mm512_slli_epi16
#define SLLI_EPI32 _mm512_slli_epi32
#define SLLV_EPI16 _mm512_sllv_epi16
#define SLLV_EPI32 _mm512_sllv_epi32
#define MAX_EPI8 _mm512_max_epi8
#define MAX_EPI16 _mm512_max_epi16
#define MAX_EPI32 _mm512_max_epi32
#define MIN_EPI8 _mm512_min_epi8
#define MIN_EPI16 _mm512_min_epi16
#define MIN_EPI32 _mm512_min_epi32

#define MADD_EPI16 _mm512_madd_epi16

#define CVTPS_EPI32 _mm512_cvtps_epi32
#define CVTEPI32_PS _mm512_cvtepi32_ps

#elif defined(__AVX__) && defined(__AVX2__) && !defined(NO_SIMD256)
#define SIMD256_AVAILABLE

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
#define MAX_PS _mm256_max_ps

#define LOAD_SI(x) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x))
#define STORE_SI(x, y) _mm256_storeu_si256(reinterpret_cast<__m256i*>(x), y)
#define SETZERO_SI _mm256_setzero_si256
#define SET1_EPI8 _mm256_set1_epi8
#define SET1_EPI16 _mm256_set1_epi16
#define SET1_EPI32 _mm256_set1_epi32
#define ADD_EPI8 _mm256_add_epi8
#define ADD_EPI16 _mm256_add_epi16
#define ADD_EPI32 _mm256_add_epi32
#define SUB_EPI8 _mm256_sub_epi8
#define SUB_EPI16 _mm256_sub_epi16
#define SUB_EPI32 _mm256_sub_epi32
#define MULLO_EPI16 _mm256_mullo_epi16
#define MULLO_EPI32 _mm256_mullo_epi32
#define AND_SI _mm256_and_si256
#define ANDNOT_SI _mm256_andnot_si256
#define OR_SI _mm256_or_si256
#define XOR_SI _mm256_xor_si256
#define SLLI_EPI16 _mm256_slli_epi16
#define SLLI_EPI32 _mm256_slli_epi32
#define SLLV_EPI16 _mm256_sllv_epi16
#define SLLV_EPI32 _mm256_sllv_epi32
#define MAX_EPI8 _mm256_max_epi8
#define MAX_EPI16 _mm256_max_epi16
#define MAX_EPI32 _mm256_max_epi32
#define MIN_EPI8 _mm256_min_epi8
#define MIN_EPI16 _mm256_min_epi16
#define MIN_EPI32 _mm256_min_epi32

#define MADD_EPI16 _mm256_madd_epi16

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
#define SET1_EPI8 _mm_set1_epi8
#define SET1_EPI16 _mm_set1_epi16
#define SET1_EPI32 _mm_set1_epi32
#define ADD_EPI8 _mm_add_epi8
#define ADD_EPI16 _mm_add_epi16
#define ADD_EPI32 _mm_add_epi32
#define SUB_EPI8 _mm_sub_epi8
#define SUB_EPI16 _mm_sub_epi16
#define SUB_EPI32 _mm_sub_epi32
#define MULLO_EPI16 _mm_mullo_epi16
#define MULLO_EPI32 _mm_mullo_epi32
#define AND_SI _mm_and_si128
#define ANDNOT_SI _mm_andnot_si128
#define OR_SI _mm_or_si128
#define XOR_SI _mm_xor_si128
#define SLLI_EPI16 _mm_slli_epi16
#define SLLI_EPI32 _mm_slli_epi32
#define SLLV_EPI16 _mm_sllv_epi16
#define SLLV_EPI32 _mm_sllv_epi32
#define MAX_EPI8 _mm_max_epi8
#define MAX_EPI16 _mm_max_epi16
#define MAX_EPI32 _mm_max_epi32
#define MIN_EPI8 _mm_min_epi8
#define MIN_EPI16 _mm_min_epi16
#define MIN_EPI32 _mm_min_epi32

#define MADD_EPI16 _mm_madd_epi16

#define CVTPS_EPI32 _mm_cvtps_epi32
#define CVTEPI32_PS _mm_cvtepi32_ps

#endif
#include <nmmintrin.h>
#include <climits>

#define PLUS(a, b) ((a) + (b))

namespace sheena{
#define UNARY_OPERATION(TARGET, LOAD, STORE, OP_NAME) {\
	for(size_t i = 0;i<size_with_padding;i+=ways){\
		STORE(TARGET + i, OP_NAME(LOAD(w + i)));\
	}\
}
#define BINARY_OPERATION(LOOP_COUNTER, TARGET, LOAD, STORE, OP_NAME, RHS) {\
	for(; LOOP_COUNTER < size_with_padding; LOOP_COUNTER += ways){\
		STORE(TARGET + LOOP_COUNTER, OP_NAME(LOAD(w + LOOP_COUNTER), RHS));\
	}\
}
#define REDUCE_OPERATION(TYPE, INIT, INIT_SIMD, LOAD, STORE, OP_SIMD, OP_SCALAR) {\
	TYPE ret = INIT;\
	size_t i = 0;\
	if(Size >= ways){\
		MM mm = INIT_SIMD;\
		if(Size >= ways * 4){\
			MM mm2 = mm, mm3 = mm, mm4 = mm;\
			const size_t e = Size - Size % (ways * 4);\
			for(;i < e;i+=ways * 4){\
				mm = OP_SIMD(LOAD(w + i), mm);\
				mm2 = OP_SIMD(LOAD(w + i + ways * 1), mm2);\
				mm3 = OP_SIMD(LOAD(w + i + ways * 2), mm3);\
				mm4 = OP_SIMD(LOAD(w + i + ways * 3), mm4);\
			}\
			mm = OP_SIMD(mm, mm2);\
			mm3 = OP_SIMD(mm3, mm4);\
			mm = OP_SIMD(mm, mm3);\
		}\
		for(;i < simd_loop_end;i+=ways){\
			mm = OP_SIMD(LOAD(w + i), mm);\
		}\
		alignas(16) TYPE v[ways];\
		STORE(v, mm);\
		if(ways == 2)ret = OP_SCALAR(v[0], v[1]);\
		else if(ways == 4)ret = OP_SCALAR(OP_SCALAR(v[0], v[1]), OP_SCALAR(v[2], v[3]));\
		else {\
			ret = OP_SCALAR(OP_SCALAR(OP_SCALAR(v[0], v[1]), OP_SCALAR(v[2], v[3])), OP_SCALAR(OP_SCALAR(v[4], v[5]), OP_SCALAR(v[6], v[7])));\
			for(int j=8;j < ways; j += 4){\
				ret = OP_SCALAR(ret, OP_SCALAR(OP_SCALAR(v[j + 0], v[j + 1]), OP_SCALAR(v[j + 2], v[j + 3])));\
			}\
		}\
		static_assert(ways == 2 || ways == 4 || ways == 8 || ways == 16 || ways == 32 || ways == 64, "");\
	}\
	if(Size != simd_loop_end){\
		for(size_t i=simd_loop_end;i<Size;i++){\
			ret = OP_SCALAR(ret, w[i]);\
		}\
	}\
	return ret;\
}
#define MATH_OPERATOR_VECTOR(VECTOR, LOAD, STORE, OP, OP_NAME)\
VECTOR operator OP(const VECTOR& rhs)const{\
	VECTOR ret;\
	size_t i = 0;\
	BINARY_OPERATION(i, ret.w, LOAD, STORE, OP_NAME, LOAD(rhs.w + i));\
	return ret;\
}\
void operator OP##=(const VECTOR& rhs){\
	size_t i = 0;\
	BINARY_OPERATION(i, w, LOAD, STORE, OP_NAME, LOAD(rhs.w + i));\
}
#define MATH_OPERATOR_SCALAR(TYPE, VECTOR, SET1, LOAD, STORE, OP, OP_NAME) \
VECTOR operator OP(TYPE rhs)const{\
	VECTOR ret;\
	MM rhs_mm = SET1(rhs);\
	size_t i = 0;\
	BINARY_OPERATION(i, ret.w, LOAD, STORE, OP_NAME, rhs_mm);\
	return ret;\
}\
void operator OP##=(TYPE rhs){\
	MM rhs_mm = SET1(rhs);\
	size_t i = 0;\
	BINARY_OPERATION(i, w, LOAD, STORE, OP_NAME, rhs_mm);\
}
#define MATH_OPERATOR(TYPE, VECTOR, SET1, LOAD, STORE, OP, OP_NAME) \
MATH_OPERATOR_VECTOR(VECTOR, LOAD, STORE, OP, OP_NAME) \
MATH_OPERATOR_SCALAR(TYPE, VECTOR, SET1, LOAD, STORE, OP, OP_NAME)

	template<size_t Size>
	class VInt;
	template<size_t Size>
	class VFlt;
	template<size_t Size>
	class alignas(Size * sizeof(float) % 64 == 0? 64 : 16) VFlt{
		friend class VInt<Size>;
#ifdef SIMD512_AVAILABLE
		static constexpr size_t ways = 64 / sizeof(float);
		using MM = __m512;
#elif defined(SIMD256_AVAILABLE)
		static constexpr size_t ways = 32 / sizeof(float);
		using MM = __m256;
#else
		static constexpr size_t ways = 16 / sizeof(float);
		using MM = __m128;
#endif
		static constexpr size_t simd_loop_end = Size - Size % ways;
		static constexpr size_t padding = Size% ways != 0 ? ways - (Size % ways) : 0;
		static constexpr size_t size_with_padding = Size + padding;
		float w[Size + padding];
#ifdef FMA_ENABLE
		static MM fma(MM v1, MM v2, MM v3){
			return FMADD_PS(v1, v2, v3);
		}
		static MM fnma(MM v1, MM v2, MM v3){
			return FNMADD_PS(v1, v2, v3);
		}
#else
		static MM fma(MM v1, MM v2, MM v3){
			return ADD_PS(MUL_PS(v1, v2), v3);
		}
		static MM fnma(MM v1, MM v2, MM v3){
			return SUB_PS(v3, MUL_PS(v1, v2));
		}
#endif
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
			UNARY_OPERATION(ret.w, LOAD_PS, STORE_PS, SQRT_PS);
			return ret;
		}
		VFlt<Size> rsqrt()const{
			VFlt<Size> ret;
			UNARY_OPERATION(ret.w, LOAD_PS, STORE_PS, RSQRT_PS);
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
			REDUCE_OPERATION(float, -FLT_MAX, SET1_PS(-FLT_MAX), LOAD_PS, STORE_PS, MAX_PS, std::max);
		}
		float min()const{
			REDUCE_OPERATION(float, FLT_MAX, SET1_PS(FLT_MAX), LOAD_PS, STORE_PS, MIN_PS, std::min);
		}
		//合計値の計算
		float sum()const{
			REDUCE_OPERATION(float, 0, SETZERO_PS(), LOAD_PS, STORE_PS, ADD_PS, PLUS);
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
				alignas(16) float v[ways];
				STORE_PS(v, mm);
#ifdef SIMD512_AVAILABLE
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
				ret += ((v[8] + v[9]) + (v[10] + v[11])) + ((v[12] + v[13]) + (v[14] + v[15]));
#elif defined(SIMD256_AVAILABLE)
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
			if(size_with_padding >= 4 * ways){
				const size_t e = size_with_padding - size_with_padding % (ways * 4);
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
		void add_product(const VFlt<Size>& v1, float x){
			size_t i = 0;
			MM x_mm = SET1_PS(x);
			if(size_with_padding >= 4 * ways){
				const size_t e = size_with_padding - size_with_padding % (ways * 4);
				for(;i<e;i+=ways * 4){
					STORE_PS(w + i, fma(LOAD_PS(v1.w + i), x_mm, LOAD_PS(w + i)));
					STORE_PS(w + i + ways * 1, fma(LOAD_PS(v1.w + i + ways * 1), x_mm, LOAD_PS(w + i + ways * 1)));
					STORE_PS(w + i + ways * 2, fma(LOAD_PS(v1.w + i + ways * 2), x_mm, LOAD_PS(w + i + ways * 2)));
					STORE_PS(w + i + ways * 3, fma(LOAD_PS(v1.w + i + ways * 3), x_mm, LOAD_PS(w + i + ways * 3)));
				}
			}
			for(;i<size_with_padding;i+=ways){
				STORE_PS(w + i, fma(LOAD_PS(v1.w + i), x_mm, LOAD_PS(w + i)));
			}
		}
		void sub_product(const VFlt<Size>& v1, const VFlt<Size>& v2){
			size_t i = 0;
			if(size_with_padding >= 4 * ways){
				const size_t e = size_with_padding - size_with_padding % (ways * 4);
				for(;i<e;i+=ways * 4){
					STORE_PS(w + i, fnma(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
					STORE_PS(w + i + ways * 1, fnma(LOAD_PS(v1.w + i + ways * 1), LOAD_PS(v2.w + i + ways * 1), LOAD_PS(w + i + ways * 1)));
					STORE_PS(w + i + ways * 2, fnma(LOAD_PS(v1.w + i + ways * 2), LOAD_PS(v2.w + i + ways * 2), LOAD_PS(w + i + ways * 2)));
					STORE_PS(w + i + ways * 3, fnma(LOAD_PS(v1.w + i + ways * 3), LOAD_PS(v2.w + i + ways * 3), LOAD_PS(w + i + ways * 3)));
				}
			}
			for(;i<size_with_padding;i+=ways){
				STORE_PS(w + i, fnma(LOAD_PS(v1.w + i), LOAD_PS(v2.w + i), LOAD_PS(w + i)));
			}
		}
		void sub_product(const VFlt<Size>& v1, float x){
			size_t i = 0;
			MM x_mm = SET1_PS(x);
			if(size_with_padding >= 4 * ways){
				const size_t e = size_with_padding - size_with_padding % (ways * 4);
				for(;i<e;i+=ways * 4){
					STORE_PS(w + i, fnma(LOAD_PS(v1.w + i), x_mm, LOAD_PS(w + i)));
					STORE_PS(w + i + ways * 1, fnma(LOAD_PS(v1.w + i + ways * 1), x_mm, LOAD_PS(w + i + ways * 1)));
					STORE_PS(w + i + ways * 2, fnma(LOAD_PS(v1.w + i + ways * 2), x_mm, LOAD_PS(w + i + ways * 2)));
					STORE_PS(w + i + ways * 3, fnma(LOAD_PS(v1.w + i + ways * 3), x_mm, LOAD_PS(w + i + ways * 3)));
				}
			}
			for(;i<size_with_padding;i+=ways){
				STORE_PS(w + i, fnma(LOAD_PS(v1.w + i), x_mm, LOAD_PS(w + i)));
			}
		}
		VInt<Size> to_vint()const;
	};

	template<size_t Size>
	class alignas(Size * sizeof(int) % 64 == 0? 64 : 16) VInt{
		friend class VFlt<Size>;
#ifdef SIMD512_AVAILABLE
		static constexpr size_t ways = 16;
		using MM = __m512i;
#elif defined(SIMD256_AVAILABLE)
		static constexpr size_t ways = 8;
		using MM = __m256i;
#else
		static constexpr size_t ways = 4;
		using MM = __m128i;
#endif
		static constexpr size_t simd_loop_end = Size - Size % ways;
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
		void operator=(const VInt<Size>& rhs){
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, LOAD_SI(rhs. w + i));
			}
		}
		int max()const{
			REDUCE_OPERATION(int, INT_MIN, SET1_EPI32(INT_MIN), LOAD_SI, STORE_SI, MAX_EPI32, std::max);
		}
		int min()const{
			REDUCE_OPERATION(int, INT_MAX, SET1_EPI32(INT_MAX), LOAD_SI, STORE_SI, MIN_EPI32, std::min);
		}
		int sum()const{
			REDUCE_OPERATION(int, 0, SETZERO_SI(), LOAD_SI, STORE_SI, ADD_EPI32, PLUS);
		}
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, +, ADD_EPI32);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, -, SUB_EPI32);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, *, MULLO_EPI32);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, &, AND_SI);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, |, OR_SI);
		MATH_OPERATOR(int, VInt<Size>, SET1_EPI32, LOAD_SI, STORE_SI, ^, XOR_SI);
		MATH_OPERATOR_VECTOR(VInt<Size>, LOAD_SI, STORE_SI, <<, SLLV_EPI32);
		VInt<Size> operator<<(int x)const{
			size_t i = 0;
			VInt<Size> ret;
			BINARY_OPERATION(i, ret.w, LOAD_SI, STORE_SI, SLLI_EPI32, x);
			return ret;
		}
		void operator<<=(int x){
			size_t i = 0;
			BINARY_OPERATION(i, w, LOAD_SI, STORE_SI, SLLI_EPI32, x);
		}
		VInt<Size> operator~()const{
			size_t i = 0;
			VInt<Size> ret;
			BINARY_OPERATION(i, ret.w, LOAD_SI, STORE_SI, ANDNOT_SI, w + i);
		}
		VFlt<Size> to_vflt()const;
	};

	template<size_t Size>
	class alignas(Size * sizeof(int16_t) % 64 == 0? 64 : 16) VInt16{
#ifdef SIMD512_AVAILABLE
		static constexpr size_t ways = 32;
		using MM = __m512i;
#elif defined(SIMD256_AVAILABLE)
		static constexpr size_t ways = 16;
		using MM = __m256i;
#else
		static constexpr size_t ways = 8;
		using MM = __m128i;
#endif
		static constexpr size_t simd_loop_end = Size - Size % ways;
		static constexpr size_t padding = Size% ways != 0 ? ways - (Size % ways) : 0;
		static constexpr size_t size_with_padding = Size + padding;
		int16_t w[size_with_padding];
	public:
		VInt16(){}
		explicit VInt16(int16_t x){
			MM x_mm = SET1_EPI16(x);
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, x_mm);
			}
		}
		VInt16(const VInt16<Size>& rhs){
			(*this) = rhs;
		}
		int16_t operator[](size_t idx)const{
			assert(idx < Size);
			return w[idx];
		}
		int16_t& operator[](size_t idx){
			assert(idx < Size);
			return w[idx];
		}
		static constexpr size_t size(){return Size;}
		void clear(){
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, SETZERO_SI());
			}
		}
		void operator=(const VInt16<Size>& rhs){
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, LOAD_SI(rhs. w + i));
			}
		}
		int16_t max()const{
			REDUCE_OPERATION(int16_t, -32768, SET1_EPI16(-32768), LOAD_SI, STORE_SI, MAX_EPI16, std::max);
		}
		int16_t min()const{
			REDUCE_OPERATION(int16_t, 32767, SET1_EPI16(32767), LOAD_SI, STORE_SI, MIN_EPI16, std::min);
		}
		int16_t sum()const{
			REDUCE_OPERATION(int16_t, 0, SETZERO_SI(), LOAD_SI, STORE_SI, ADD_EPI16, PLUS);
		}
		MATH_OPERATOR(int16_t, VInt16<Size>, SET1_EPI16, LOAD_SI, STORE_SI, +, ADD_EPI16);
		MATH_OPERATOR(int16_t, VInt16<Size>, SET1_EPI16, LOAD_SI, STORE_SI, -, SUB_EPI16);
		MATH_OPERATOR(int16_t, VInt16<Size>, SET1_EPI16, LOAD_SI, STORE_SI, *, MULLO_EPI16);
		MATH_OPERATOR(int16_t, VInt16<Size>, SET1_EPI16, LOAD_SI, STORE_SI, &, AND_SI);
		MATH_OPERATOR(int16_t, VInt16<Size>, SET1_EPI16, LOAD_SI, STORE_SI, |, OR_SI);
		MATH_OPERATOR(int16_t, VInt16<Size>, SET1_EPI16, LOAD_SI, STORE_SI, ^, XOR_SI);
		MATH_OPERATOR_VECTOR(VInt16<Size>, LOAD_SI, STORE_SI, <<, SLLV_EPI32);
		VInt16<Size> operator<<(int x)const{
			size_t i = 0;
			VInt16<Size> ret;
			BINARY_OPERATION(i, ret.w, LOAD_SI, STORE_SI, SLLI_EPI16, x);
			return ret;
		}
		void operator<<=(int x){
			size_t i = 0;
			BINARY_OPERATION(i, w, LOAD_SI, STORE_SI, SLLI_EPI16, x);
		}
		VInt16<Size> operator~()const{
			size_t i = 0;
			VInt16<Size> ret;
			BINARY_OPERATION(i, ret.w, LOAD_SI, STORE_SI, ANDNOT_SI, w + i);
			return ret;
		}
		//内積計算
		int32_t inner_product(const VInt16<Size>& rhs)const{
			int32_t ret = 0;
			if(Size >= ways){
				MM mm1 = SETZERO_SI();
				//インライン展開
				size_t i = 0;
				if(Size >= ways * 4){
					MM mm2 = SETZERO_SI(), mm3 = SETZERO_SI(), mm4 = SETZERO_SI();
					const size_t e = Size - Size % (ways * 4);
					for(;i < e;i+=ways * 4){
						mm1 = ADD_EPI32(MADD_EPI16(LOAD_SI(w + i + ways * 0), LOAD_SI(rhs.w + i + ways * 0)), mm1);
						mm2 = ADD_EPI32(MADD_EPI16(LOAD_SI(w + i + ways * 1), LOAD_SI(rhs.w + i + ways * 1)), mm2);
						mm3 = ADD_EPI32(MADD_EPI16(LOAD_SI(w + i + ways * 2), LOAD_SI(rhs.w + i + ways * 2)), mm3);
						mm4 = ADD_EPI32(MADD_EPI16(LOAD_SI(w + i + ways * 3), LOAD_SI(rhs.w + i + ways * 3)), mm4);
					}
					mm1 = ADD_EPI32(mm1, mm2);
					mm3 = ADD_EPI32(mm3, mm4);
					mm1 = ADD_EPI32(mm1, mm3);
				}
				if(Size % (ways * 4) != 0){
					for(;i < simd_loop_end;i+=ways){
						mm1 = ADD_EPI32(mm1, MADD_EPI16(LOAD_SI(w + i), LOAD_SI(rhs.w + i)));
					}
				}
				alignas(16) int32_t v[ways / 2];
				STORE_SI(v, mm1);
#ifdef SIMD512_AVAILABLE
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
				ret += ((v[8] + v[9]) + (v[10] + v[11])) + ((v[12] + v[13]) + (v[14] + v[15]));
#elif defined(SIMD256_AVAILABLE)
				ret = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
#else
				ret = (v[0] + v[1]) + (v[2] + v[3]);
#endif
			}
			if(Size != simd_loop_end){
				for(size_t i=simd_loop_end;i<Size;i++){
					ret += int(w[i]) * int(rhs.w[i]);
				}
			}
			return ret;
		}
	};

	template<size_t Size>
	class alignas(Size * sizeof(int8_t) % 64 == 0? 64 : 16) VInt8{
#ifdef SIMD512_AVAILABLE
		static constexpr size_t ways = 64;
		using MM = __m512i;
#elif defined(SIMD256_AVAILABLE)
		static constexpr size_t ways = 32;
		using MM = __m256i;
#else
		static constexpr size_t ways = 16;
		using MM = __m128i;
#endif
		static constexpr size_t simd_loop_end = Size - Size % ways;
		static constexpr size_t padding = Size% ways != 0 ? ways - (Size % ways) : 0;
		static constexpr size_t size_with_padding = Size + padding;
		int8_t w[size_with_padding];
	public:
		VInt8(){}
		explicit VInt8(int8_t x){
			MM x_mm = SET1_EPI8(x);
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, x_mm);
			}
		}
		VInt8(const VInt8<Size>& rhs){
			(*this) = rhs;
		}
		int8_t operator[](size_t idx)const{
			assert(idx < Size);
			return w[idx];
		}
		int8_t& operator[](size_t idx){
			assert(idx < Size);
			return w[idx];
		}
		static constexpr size_t size(){return Size;}
		void clear(){
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, SETZERO_SI());
			}
		}
		void operator=(const VInt8<Size>& rhs){
			for(size_t i=0;i<size_with_padding;i+=ways){
				STORE_SI(w + i, LOAD_SI(rhs. w + i));
			}
		}
		int8_t max()const{
			REDUCE_OPERATION(int8_t, -128, SET1_EPI8(-128), LOAD_SI, STORE_SI, MAX_EPI8, std::max);
		}
		int8_t min()const{
			REDUCE_OPERATION(int8_t, 127, SET1_EPI8(127), LOAD_SI, STORE_SI, MIN_EPI8, std::min);
		}
		MATH_OPERATOR(int8_t, VInt8<Size>, SET1_EPI8, LOAD_SI, STORE_SI, +, ADD_EPI8);
		MATH_OPERATOR(int8_t, VInt8<Size>, SET1_EPI8, LOAD_SI, STORE_SI, -, SUB_EPI8);
		MATH_OPERATOR(int8_t, VInt8<Size>, SET1_EPI8, LOAD_SI, STORE_SI, &, AND_SI);
		MATH_OPERATOR(int8_t, VInt8<Size>, SET1_EPI8, LOAD_SI, STORE_SI, |, OR_SI);
		MATH_OPERATOR(int8_t, VInt8<Size>, SET1_EPI8, LOAD_SI, STORE_SI, ^, XOR_SI);
		VInt8<Size> operator~()const{
			size_t i = 0;
			VInt8<Size> ret;
			BINARY_OPERATION(i, ret.w, LOAD_SI, STORE_SI, ANDNOT_SI, w + i);
		}
	};
	template<size_t Size>
	VInt<Size> VFlt<Size>::to_vint()const{
		VInt<Size> ret;
		UNARY_OPERATION(ret.w, LOAD_PS, STORE_SI, CVTPS_EPI32);
		return ret;
	}
	template<size_t Size>
	VFlt<Size> VInt<Size>::to_vflt()const{
		VFlt<Size> ret;
		UNARY_OPERATION(ret.w,LOAD_SI, STORE_PS, CVTEPI32_PS);
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
#undef SQRT_PS
#undef RSQRT_PS
#undef MIN_PS
#undef MAX_PS

#undef LOAD_SI
#undef STORE_SI
#undef SETZERO_SI
#undef SET1_EPI8
#undef SET1_EPI16
#undef SET1_EPI32
#undef ADD_EPI8
#undef ADD_EPI16
#undef ADD_EPI32
#undef SUB_EPI8
#undef SUB_EPI16
#undef SUB_EPI32
#undef MULLO_EPI16
#undef MULLO_EPI32
#undef AND_SI
#undef ANDNOT_SI
#undef OR_SI
#undef XOR_SI
#undef SLLI_EPI16
#undef SLLI_EPI32
#undef SLLV_EPI16
#undef SLLV_EPI32
#undef MAX_EPI8
#undef MAX_EPI16
#undef MAX_EPI32
#undef MIN_EPI8
#undef MIN_EPI16
#undef MIN_EPI32

#undef MADD_EPI16

#undef CVTPS_EPI32
#undef CVTEPI32_PS

#undef PLUS

#undef MATH_OPERATOR
#undef MATH_OPERATOR_SCALAR
#undef MATH_OPERATOR_VECTOR
#undef UNARY_OPERATION
#undef BINARY_OPERATION
#undef REDUCE_OPERATION

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
