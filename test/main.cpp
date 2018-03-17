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
		std::cout << "test is failed." << err_msg << std::endl;
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
static void test_mcts();
class State;
int main(void){
	test_common();
	test_math();
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
class State{
	using Action = int;
	static constexpr size_t NPlayer = 1;
	static constexpr size_t ActionDim = 4;
	mutable std::mt19937 mt;
	int number;
	int turn;
public:
	void playout(sheena::Array<double, NPlayer>& reward){
		while(number < 40){
			act(mt() % ActionDim + 1);
		}
		reward[0] = 10.0 / turn;
	}
	int get_actions(int& n, sheena::Array<Action, ActionDim>& actions, sheena::Array<float, ActionDim>& p)const{
		if(number >= 40){
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
	State():mt(0), number(0), turn(0){}
	State(const State& state):mt(state.mt()), number(state.number), turn(state.turn){}
};
static void test_mcts(){
	sheena::mcts::Searcher<State, int, 1, 4> searcher;
	State state;
	searcher.set_C(5.0);
	searcher.search(state, 100000);
	sheena::Array<int, 4> actions;
	sheena::Array<double, 4> rewards;
	sheena::Array<int, 4> count;
	int n = searcher.search_result(state, actions, rewards, count);
	ok_if_true(n == 4);
	for(int i=0;i<n;i++){
		std::cout << actions[i] << " " << rewards[i] << " " << count[i] << std::endl;
	}
}