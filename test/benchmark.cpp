#include "../sheena.hpp"

#include <random>
#include <iostream>

std::mt19937 mt;
std::normal_distribution<double> dist;
static float result;
template<size_t size>
void ip_test(){
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
	std::cout << "size = " << size << " " << msec << "[msec]" << double(loop * size) * 2 / msec / 1000 / 1000 << "G flops" << std::endl;
	return;
}
int main(void){
	ip_test<512>();
	ip_test<1024>();
	ip_test<2048>();
	ip_test<4096>();
	ip_test<8192>();
	return 0;
}