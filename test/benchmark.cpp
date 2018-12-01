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
static void add_test(){
	sheena::VFlt<size> v1, v2;
	for(int i=0;i<size;i++){
		v1[i] = dist(mt);
		v2[i] = dist(mt);
	}
	constexpr size_t loop = (2ULL << 34) / size;
	sheena::Stopwatch stopwatch;
	for(size_t i=0; i<loop;i++){
		v1 += v2;
	}
	uint64_t msec = stopwatch.msec();
	std::cout << "add : size = " << size << " " << msec << "[msec]" << double(loop * size) / msec / 1000 / 1000 << "G flops" << std::endl;
}
template<size_t size>
static void bench(){
	ip_test<size>();
	add_test<size>();
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
	return 0;
}