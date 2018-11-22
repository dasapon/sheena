#include "../sheena.hpp"
#include <thread>
#include <iostream>
#include <random>
#include <string>
#define ok_if_true(boolean){\
	my_assert(boolean, std::string("test is failed.") + std::to_string(__LINE__));\
}
#define ok_if_equal(v1, v2){\
	my_assert_fp(v1, v2, std::string("test is failed.") + std::to_string(__LINE__));\
}

static void my_assert(bool eq, std::string err_msg){
	if(!eq){
		std::cout << err_msg << std::endl;
	}
}
static void my_assert_fp(double v1, double v2, std::string err_msg){
	double diff = std::abs(v1 - v2);
	constexpr double allowable = 0.001;
	if(diff > allowable && diff > std::abs(v1) * allowable && diff > std::abs(v2) * allowable){
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
template<typename VFLT, typename Function>
static void fp_vector_test(const VFLT& result, const VFLT& origin, Function f){
	for(size_t i=0;i<VFLT::size();i++){
		ok_if_equal(result[i], f(origin[i]));
	}
}
template<typename Vfp>
static void test_simd_fp(){
	Vfp v;
	for(int i=0;i<Vfp::size();i++){
		v[i] = i + 1;
	}
	Vfp v2(v);
	ok_if_equal(v2.sum(), (1 + Vfp::size()) * Vfp::size() / 2);
	ok_if_equal(v2.max(), Vfp::size());
	ok_if_equal(v2.min(), 1);
	v = v2;
	fp_vector_test(v, v2, [](double d){
		return d;
	});
	v += v2;
	fp_vector_test(v, v2, [](double d){
		return d + d;
	});
	v = v2 + v2;
	fp_vector_test(v, v2, [](double d){
		return d + d;
	});
	v = v2;
	v -= v2;
	fp_vector_test(v, v2, [](double d){
		return d - d;
	});
	v = v2 - v2;
	fp_vector_test(v, v2, [](double d){
		return d - d;
	});
	v = v2;
	v *= v2;
	fp_vector_test(v, v2, [](double d){
		return d * d;
	});
	v = v2 * v2;
	fp_vector_test(v, v2, [](double d){
		return d * d;
	});
	v = v2;
	v /= v2;
	fp_vector_test(v, v2, [](double d){
		return d / d;
	});
	v = v2 / v2;
	fp_vector_test(v, v2, [](double d){
		return d / d;
	});
	v = v2.sqrt();
	fp_vector_test(v, v2, [](double d){
		return std::sqrt(d);
	});
	v = v2.rsqrt();
	fp_vector_test(v, v2, [](double d){
		return 1.0 / std::sqrt(d);
	});
	double ip = v2.inner_product(v2);
	float ip_ = 0;
	for(size_t i=0;i<Vfp::size();i++)ip_ += (i + 1) * (i + 1);
	ok_if_equal(ip, ip_);
	//スカラーとの演算のテスト
	for(size_t i=0;i<Vfp::size();i++){
		v[i] = i + 1;
	}
	v2 = v + 1;
	v += 1;
	for(size_t i=0;i<Vfp::size();i++){
		ok_if_equal(v[i], v2[i]);
		ok_if_equal(v[i], i + 2.0f);
	}
	v2 = v - 1;
	v -= 1;
	for(size_t i=0;i<Vfp::size();i++){
		ok_if_equal(v[i], v2[i]);
		ok_if_equal(v[i], i + 1.0f);
	}
	v2 = v * 2;
	v *= 2;
	for(size_t i=0;i<Vfp::size();i++){
		ok_if_equal(v[i], v2[i]);
		ok_if_equal(v[i], (i + 1) * 2.0f);
	}
	v2 = v / 2;
	v /= 2;
	for(size_t i=0;i<Vfp::size();i++){
		ok_if_equal(v[i], v2[i]);
		ok_if_equal(v[i], i + 1.0f);
	}
}

template<typename Ty, typename VI>
static void test_simd_int(){
	VI vi;
	for(size_t i = 0; i<VI::size();i++)vi[i] = i;
	Ty max = vi[0], min = vi[0];
	for(size_t i = 1; i<VI::size();i++){
		max = std::max(vi[i], max);
		min = std::min(vi[i], min);
	}
	ok_if_true(vi.max() == max);
	ok_if_true(vi.min() == min);
	if(vi.min() != min){
		std::cout << vi.size() << "," << int64_t(min) << "," << int64_t(vi.min()) << "," << sizeof(Ty) << std::endl;
	}
	VI vi2(vi);
	vi += vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] + vi2[i]));
	}
	vi = vi2 + vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] + vi2[i]));
	}
	vi = vi2;
	vi -= vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] - vi2[i]));
	}
	vi = vi2 - vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] - vi2[i]));
	}
	vi = vi2;
	vi |= vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] | vi2[i]));
	}
	vi = vi2 | vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] | vi2[i]));
	}
	vi = vi2;
	vi &= vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] & vi2[i]));
	}
	vi = vi2 & vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] & vi2[i]));
	}
	vi = vi2;
	vi ^= vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] ^ vi2[i]));
	}
	vi = vi2 ^ vi2;
	for(size_t i=0;i<VI::size();i++){
		ok_if_true(vi[i] == static_cast<Ty>(vi2[i] ^ vi2[i]));
	}
}
template<size_t Size>
static void test_simd_sub(){
	test_simd_fp<sheena::VFlt<Size>>();

	//整数演算のテスト
	test_simd_int<int32_t, sheena::VInt<Size>>();
	test_simd_int<int16_t, sheena::VInt16<Size>>();
	test_simd_int<int8_t, sheena::VInt8<Size>>();
	sheena::VInt<Size> vi;
	for(size_t i=0;i<Size;i++)vi[i] = i;
	ok_if_true(vi.max() == Size - 1);
	ok_if_true(vi.min() == 0);
	ok_if_true(vi.sum() == (Size - 1) * Size / 2);
	sheena::VInt<Size> vi2(vi);
	vi *= vi2;
	for(size_t i=0;i<Size;i++){
		ok_if_true(vi[i] == int(i) * int(i));
	}
	//変換のテスト
	sheena::VFlt<Size> vflt;
	for(int i=0;i<Size;i++)vflt[i] = i -2;
	vi = vflt.to_vint();
	for(size_t i=0;i<Size;i++){
		ok_if_equal(float(vi[i]), vflt[i]);
	}
	for(int i=0;i<Size;i++){
		vi[i] = -i;
	}
	vflt = vi.to_vflt();
	for(size_t i=0;i<Size;i++){
		ok_if_equal(float(vi[i]), vflt[i]);
	}
}
static void test_simd(){
	test_simd_sub<3>();
	test_simd_sub<6>();
	test_simd_sub<9>();
	test_simd_sub<12>();
	test_simd_sub<15>();
	test_simd_sub<18>();
	test_simd_sub<1625>();
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
		while(!terminate(reward, thread_id)){
			act(mt[thread_id]() % ActionDim + 1, thread_id);
		}
	}
	bool terminate(sheena::Array<double, NPlayer>& reward, size_t thread_id)const{
		if(number >= 80){
			reward[0] = 20.0 / turn;
			return true;
		}
		return false;
	}
	int get_actions(int& n, sheena::Array<Action, ActionDim>& actions, sheena::Array<float, ActionDim>& p, size_t thread_id)const{
		if(number >= 80){
			n = 0;
			return 0;
		}
		n = ActionDim;
		for(size_t i=0;i<ActionDim;i++){
			actions[i] = i + 1;
			p[i] = 1.0 / ActionDim;
		}
		return 0;
	}
	uint64_t key()const{return turn * 40 + number;}
	void act(Action a, size_t thread_id){
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
	searcher.search(state, 1000, 1000000);
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