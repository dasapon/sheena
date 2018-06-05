#include "../sheena.hpp"
#include <thread>
#include <iostream>
#include <random>
#include <string>
#define ok_if_true(boolean){\
	my_assert(boolean, std::string("test is failed.") + std::to_string(__LINE__));\
}
#define ok_if_equal(v1, v2){\
	my_assert_float(v1, v2, std::string("test is failed.") + std::to_string(__LINE__));\
}

static void my_assert(bool eq, std::string err_msg){
	if(!eq){
		std::cout << err_msg << std::endl;
	}
}
template<typename Ty>
static void my_assert_float(Ty v1, Ty v2, std::string err_msg){
	Ty diff = std::abs(v1 - v2);
	constexpr Ty allowable = 0.0001;
	if(diff > allowable && diff > std::abs(v1) * allowable && diff < std::abs(v2) * allowable){
		std::cout << err_msg << " " << v1 << " " << v2 << std::endl;
	}
}
static void test_common();
static void test_math();
static void test_simd();
static void test_mcts();

class State;
int main(void){
	std::cout << "common" << std::endl;
	test_common();
	std::cout << "math" << std::endl;
	test_math();
	std::cout << "simd" << std::endl;
	test_simd();
	std::cout << "mcts" << std::endl;
	test_mcts();
	std::cout << "test is end" << std::endl;
	return 0;
}
static void test_common(){
	std::vector<int> v({2, 5, 2, 6});
	sheena::remove(v, [](int x){return x == 2;});
	sheena::Array3d<int, 2, 2, 2> array3d;
	ok_if_true(v.size() == 2);
	ok_if_true(v[0] == 5);
	ok_if_true(v[1] == 6);
}
static void test_math(){
	//mathの単体テスト
	ok_if_equal(0.5, sheena::sigmoid(0));
	ok_if_true(sheena::sigmoid(1) > 0.5);
	ok_if_true(sheena::sigmoid(-1) < 0.5);
	//softmax
	sheena::Array<float, 2> array({0, 0});
	sheena::ArrayAlloc<float> alloc(2);
	std::normal_distribution<float> dist(0.0, 1.0);
	std::mt19937 mt;
	for(int i=0;i<1000;i++){
		alloc[0] = i;//null pointerで落ちないか?
		ok_if_true(alloc[0] == i);
		array[0] = dist(mt);
		array[1] = dist(mt);
		auto probability = array;
		sheena::softmax<2>(probability, 2);
		ok_if_true((array[0] >= array[1]) == (probability[0] >= probability[1]));
	}
}

template<size_t Size>
static void test_simd_sub(){
	//浮動小数点演算のテスト
	sheena::VFlt<Size> v;
	for(int i=0;i<Size;i++){
		v[i] = i;
	}
	
	sheena::VFlt<Size> v2(v);
	v += v2;
	for(int i=0;i<Size;i++){
		ok_if_equal(v[i], float(i + i));
	}
	float ip = v2.inner_product(v2);
	float ip_ = 0;
	for(int i=0;i<Size;i++)ip_ += i * i;
	ok_if_equal(ip, ip_);
	//整数演算のテスト
	sheena::VInt<Size> vi;
	for(int i=0;i<Size;i++)vi[i] = i;
	sheena::VInt<Size> vi2(vi);
	vi *= vi2;
	for(int i=0;i<Size;i++){
		ok_if_true(vi[i] == i * i);
	}
}
static void test_simd(){
	test_simd_sub<3>();
	test_simd_sub<6>();
	test_simd_sub<9>();
	test_simd_sub<12>();
}
class State{
	using Action = int;
	static constexpr size_t NPlayer = 1;
	static constexpr size_t ActionDim = 4;
	static sheena::Array<std::mt19937, 2> mt;
	int number;
	int turn;
public:
	void playout(sheena::Array<double, NPlayer>& reward, size_t thread_id){
		while(number < 80){
			act(mt[thread_id]() % ActionDim + 1);
		}
		reward[0] = 20.0 / turn;
	}
	int get_actions(int& n, sheena::Array<Action, ActionDim>& actions, sheena::Array<float, ActionDim>& p, size_t thread_id)const{
		if(number >= 80){
			n = 0;
			return 0;
		}
		n = ActionDim;
		for(int i=0;i<ActionDim;i++){
			actions[i] = i + 1;
			p[i] = 1.0 / ActionDim;
		}
		return 0;
	}
	uint64_t key()const{return turn * 40 + number;}
	void act(Action a){
		number += a;
		turn++;
	}
	State():number(0), turn(0){}
	State(const State& state):number(state.number), turn(state.turn){}
};
sheena::Array<std::mt19937, 2> State::mt;
static void test_mcts(){
	sheena::mcts::Searcher<sheena::mcts::UCB1, State, int, 1, 4> searcher;
	State state;
	searcher.set_C(1.4);
	searcher.set_threads(1);
	sheena::Stopwatch stopwatch;
	searcher.search(state, 50000, 1000000);
	sheena::Array<int, 4> actions;
	sheena::Array<double, 4> rewards;
	sheena::Array<int, 4> count;
	int n = searcher.search_result(state, actions, rewards, count);
	ok_if_true(n == 4);
	for(int i=0;i<n;i++){
		std::cout << actions[i] << " " << rewards[i] << " " << count[i] << std::endl;
	}
	std::cout << stopwatch.msec() << "[ms]" << std::endl;
}