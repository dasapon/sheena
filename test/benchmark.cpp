#include "../sheena.hpp"

#include <random>
#include <iostream>

std::mt19937 mt;
std::normal_distribution<double> dist;
template<size_t size>
static void ip_test(){
	double result = 0;
	sheena::VFlt<size> v1, v2;
	for(int i=0;i<size;i++){
		v1[i] = dist(mt);
		v2[i] = dist(mt);
	}
	constexpr size_t loop = (2ULL << 34) / size;
	sheena::Stopwatch stopwatch;
	for(size_t i=0; i<loop;i++){
		result += v1.inner_product(v2);
	}
	uint64_t msec = stopwatch.msec();
	std::cout << "inner product : " << result / loop << std::endl;
	std::cout << " size = " << size << " " << msec << "[msec] " << double(loop * size) * 2 / msec / 1000 / 1000 << " Gflops" << std::endl;
	return;
}
template<size_t size>
static void exp_test(){
	sheena::VFlt<size> v1, v2, va, vb;
	for(int i=0;i<size;i++){
		v1[i] = dist(mt);
	}
	v2 = v1;
	sheena::Stopwatch stopwatch;
	for(int i=0;i<10000; i++){
		if(mt() & 1){
			va = v2.exp();
		}
		else{
			vb = v2.exp();
		}
	}
	uint64_t msec = stopwatch.msec();
	std::cout << msec << "[msec]" << std::endl;
	stopwatch.restart();
	for(int rep=0;rep<10000; rep++){
		if(mt() & 1){
			for(int i=0;i<size;i++){
				va[i] = std::exp(v1[i]);
			}
		}
		else{
			for(int i=0;i<size;i++){
				vb[i] = std::exp(v1[i]);
			}
		}
	}
	msec = stopwatch.msec();
	std::cout << msec << "[msec]" << std::endl;
	va = v2.exp();
	double sq_err = 0;
	for(int i=0;i<size;i++){
		float y = std::exp(v2[i]);
		sq_err += (y - va[i]) * (y - va[i]);
	}
	std::cout << sq_err / size << std::endl;
}
template<size_t size>
static void add_sub_test(){
	sheena::VFlt<size> v1, v2;//, v3;
	for(int i=0;i<size;i++){
		v1[i] = dist(mt);
		v2[i] = dist(mt);
		//v3[i] = dist(mt);
	}
	constexpr size_t loop = (2ULL << 34) / size;
	sheena::Stopwatch stopwatch;
	for(size_t i=0; i<loop;i++){
		v1 += v2;
		//v1 -= v2;
	}
	uint64_t msec = stopwatch.msec();
	std::cout << "add_sub : size = " << size << " " << msec << "[msec]" << double(loop * size) / msec / 1000 / 1000 << "G flops" << std::endl;
}
template<size_t size>
static void bench(){
	ip_test<size>();
	add_sub_test<size>();
}
int main(void){
	//todo : ベンチマークの結果がおかしい(size 64, 128, 256で理論性能を超える)ので原因を調べる
	bench<16>();
	bench<32>();
	bench<64>();
	bench<128>();
	bench<256>();
	bench<512>();
	bench<1024>();
	bench<2048>();
	bench<4096>();
	bench<8192>();
	exp_test<65536>();
	return 0;
}