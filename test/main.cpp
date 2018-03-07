#include "../sheena.hpp"
#include <thread>
#include <iostream>
#define sheena_assert(boolean){\
	if(!(boolean)){\
		std::cout << "test_is failed. " << __LINE__ << std::endl;\
		return -1;\
	}\
}
int main(void){
	std::vector<int> v({2, 5, 2, 6});
	sheena::remove(v, [](int x){return x == 2;});
	sheena_assert(v.size() == 2);
	sheena_assert(v[0] == 5);
	sheena_assert(v[1] == 6);
	std::cout << "test is succeeded" << std::endl;
	return 0;
}