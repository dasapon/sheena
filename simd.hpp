#pragma once

#include <xmmintrin.h>
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
}