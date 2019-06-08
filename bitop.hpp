#pragma once

#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace sheena{
	inline int popcnt64(uint64_t x){
#ifdef _MSC_VER
		return static_cast<int>(__popcnt64(x));
#elif defined(__x86_64__)
		return __builtin_popcountll(x);
#else
		static_assert(false);
#endif
	}
	inline int popcnt32(uint32_t x){
#ifdef _MSC_VER
		return static_cast<int>(__popcnt(x));
#elif defined(__x86_64__)
		return __builtin_popcount(x);
#else
		static_assert(false);
#endif
	}
	inline int bsf64(uint64_t x){
#ifdef _MSC_VER
		unsigned long ret = 0;
		_BitScanForward64(&ret, x);
#elif defined(__x86_64__)
		int ret = __builtin_ctzll(x);
#else
		static_assert(false);
#endif
		return ret;
	}
	inline int bsf32(uint32_t x){
#ifdef _MSC_VER
		unsigned long ret = 0;
		_BitScanForward(&ret, x);
#elif defined(__x86_64__)
		int ret = __builtin_ctz(x);
#else
		static_assert(false);
#endif
		return ret;
	}
}