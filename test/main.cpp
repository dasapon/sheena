#include "../sheena.hpp"
#include <thread>
#include <iostream>
#include <random>
#include <string>
#define ok_if_true(boolean){\
	my_assert(boolean, std::string("test is failed.") + std::to_string(__LINE__));\
}

static void my_assert(bool eq, std::string err_msg){
	if(!eq){
		std::cout << "test is failed." << err_msg << std::endl;
		std::exit(-1);
	}
}
static void my_assert_float(float v1, float v2, std::string err_msg){
	float diff = std::abs(v1 - v2);
	constexpr float allowable = 0.0001;
	if(diff > allowable && diff > std::abs(v1) * allowable && diff < std::abs(v2) * allowable){
		std::cout << err_msg << " " << v1 << " " << v2 << std::endl;
	}
}
static void test_common();
int main(void){
	test_common();
	std::cout << "test is succeeded" << std::endl;
	return 0;
}
static void test_common(){
	std::vector<int> v({2, 5, 2, 6});
	sheena::remove(v, [](int x){return x == 2;});
	ok_if_true(v.size() == 2);
	ok_if_true(v[0] == 5);
	ok_if_true(v[1] == 6);
}