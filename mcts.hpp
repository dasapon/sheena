
#include <cmath>
#include <mutex>
#include <thread>

#include "common.hpp"

namespace sheena::mcts{
	//汎用のMCTSを実現する
	//Stateに実装される必要のあるメソッド
	//コピーコンストラクタ
	//State(const State&)
	//actionを実行し,状態遷移する
	//void act(Action)
	//playoutを行い, 報酬を得る
	//thread_idはそのplayoutを呼び出したスレッドの番号
	//void playout(sheena::Array<double, NPlayer>&, size_t thread_id)
	//そのStateにおける行動とそれらの事前確率を得る
	//戻り値は手番プレイヤの番号(0, 1, ... NPlayer-1)
	//また、ゲーム終了時には可能な行動数を0として返す
	//int get_actions(int&, sheena::Array<Action, MaxAction>&, sheena::Array<float, MaxAction>&)const;
	//局面のハッシュ値を返す
	//uint64_t key()const;
	enum MCTS_TYPE{
		UCB1,
		PUCT,
	};
	template<MCTS_TYPE type, typename State, typename Action, size_t NPlayer, size_t MaxAction>
	class Searcher{
		static constexpr int chain_size = 4;
		double explore_coefficient;
		int expansion_threshold;
		int virtual_loss;
		ArrayAlloc<std::thread> threads;
	public:
	private:
		struct Edge{
			double reward;
			uint64_t played;
			Edge():reward(0), played(0){}
			void update(double r, int vl){
				reward += r;
				played += 1 - vl;
			}
			double Q()const{
				if(played == 0)return 0;
				else return reward / played;
			}
		};
		class Node{
			uint64_t key_;
			int generation_;
			int turn_player;
			int n_action;
			int total_played;
			sheena::Array<Action, MaxAction> actions;
			sheena::Array<float, MaxAction> prior_probability;
			sheena::Array<Edge, MaxAction> edges;
		public:
			Node():generation_(-1), n_action(0){}
			void invalidate(uint64_t idx){
				generation_ = -1;
				//keyが合わないようにする
				key_ = idx + 1;
			}
			uint64_t key()const{return key_;}
			int generation()const{return generation_;}
			void update_generation(int gen){generation_ = gen;}
			bool set_up(const State& state, int gen){
				generation_ = gen;
				key_ = state.key();
				total_played = 0;
				turn_player = state.get_actions(n_action, actions, prior_probability);
				if(n_action == 0){
					//終端ノードは置換表には不要
					invalidate(key_);
					return false;
				}
				return true;
			}
			void update(int act_idx, Array<double, NPlayer>& reward, int vl){
				total_played += 1 - vl;
				edges[act_idx].update(reward[turn_player], vl);
			}
			Action select(int& idx, bool& expand, double C, int exp_th, int vl){
				double max_score = -DBL_MAX;
				double expl;
				switch(type){
				case UCB1:
					expl = std::log(double(total_played));
					break;
				case PUCT:
					expl = C * std::sqrt(double(total_played));
					break;
				}
				for(int i=0;i<n_action;i++){
					double q = edges[i].Q();
					double score;
					switch(type){
					case UCB1:
						if(edges[i].played == 0)score = 1000000 + prior_probability[i];
						else score = q + C * std::sqrt(expl / (edges[i].played));
						break;
					case PUCT:
						if(total_played == 0)score = prior_probability[i];
						else score = q + prior_probability[i] * expl / (edges[i].played + 1);
						break;
					}
					if(max_score < score){
						idx = i;
						max_score = score;
					}
				}
				expand = edges[idx].played > exp_th;
				//virtual loss
				edges[idx].played += vl;
				total_played+=vl;
				return actions[idx];
			}
			int search_result(Array<Action, MaxAction>& actions, Array<double, MaxAction>& rewards, 
			Array<int, MaxAction>& count)const{
				int total = 0;
				for(int i=0;i<n_action;i++){
					actions[i] = this->actions[i];
					rewards[i] = this->edges[i].Q();
					count[i] = this->edges[i].played;
					total += count[i];
				}
				assert(total == total_played);
				return n_action;
			}
		};
		ArrayAlloc<std::pair<Array<Node, chain_size>, std::mutex>> tt;
		int generation_;
		Node* get_node(State& state, bool expand){
			uint64_t key = state.key();
			size_t idx = key % tt.size();
			int empty = -1;
			for(int i=0;i<chain_size;i++){
				if(tt[idx].first[i].key() == key){
					tt[idx].first[i].update_generation(generation_);
					return &tt[idx].first[i];
				}
				if(tt[idx].first[i].generation() != generation_)empty = i;
			}
			//開いている箇所を使ってノード展開
			if(empty != -1 && expand){
				if(tt[idx].first[empty].set_up(state, generation_))return &tt[idx].first[empty];
			}
			return nullptr;
		}
		void search_rec(State& state, Array<double, NPlayer>& reward, bool expand, size_t thread_id);
	public:
		Searcher():explore_coefficient(1.0), expansion_threshold(0), virtual_loss(3), threads(1), tt(16384){
			generation_ = 1;
			clear_tt();
		}
		void search(const State& root, size_t time_limit, size_t po){
			generation_++;
			if(generation_ < 0)generation_ = 1;
			Stopwatch stopwatch;
			auto proce = [=](size_t thread_id, size_t t, size_t p){
				sheena::Array<double, NPlayer> reward;
				for(int i=0;i<p;i++){
					State state(root);
					search_rec(state, reward, true, thread_id);
					if(stopwatch.msec() >= t)break;
				}
			};
			for(int i=0;i<threads.size();i++){
				size_t p = po / threads.size();
				if(po % threads.size() < i)p++;
				threads[i] = std::thread(proce, i, time_limit, p);
			}
			for(int i=0;i<threads.size();i++){
				threads[i].join();
			}
		}
		void clear_tt(){
			for(size_t i=0;i<tt.size();i++){
				for(int j=0;j<chain_size;j++)tt[i].first[j].invalidate(i);
			}
		}
		void resize_tt(size_t sz){
			tt.resize(sz / chain_size);
			clear_tt();
		}
		void set_C(double c){
			if(c <= 0)throw std::invalid_argument("");
			explore_coefficient = c;
		}
		void set_expansion_threshold(int X){
			if(X<0)throw std::invalid_argument("");
			expansion_threshold = X;
		}
		void set_virtual_loss(int X){
			if(X < 0)throw std::invalid_argument("");
			virtual_loss = X;
		}
		void set_threads(size_t X){
			if(X <= 0)throw std::invalid_argument("");
			threads.resize(X);
		}
		//stateにおける各行動の過去の報酬の平均と、play回数を得る
		//戻り値は合法手数(そもそもそのstateがtree中に無い場合は-1)
		int search_result(const State& state, Array<Action, MaxAction>& actions, 
		Array<double, MaxAction>& rewards, Array<int, MaxAction>& count)const{
			uint64_t key = state.key();
			size_t idx = key % tt.size();
			for(int i=0;i<chain_size;i++){
				if(tt[idx].first[i].key() == key){
					return tt[idx].first[i].search_result(actions, rewards, count);
				}
			}
			return -1;
		}
	};
	template<MCTS_TYPE type, typename State, typename Action, size_t NPlayer, size_t MaxAction>
	void Searcher<type, State, Action, NPlayer, MaxAction>::search_rec(
		State& state, Array<double, NPlayer>& reward, bool expand, size_t thread_id){
		uint64_t tt_idx = state.key() % tt.size();
		//ロックをかけ, Nodeを取得
		std::unique_lock<std::mutex> lock(tt[tt_idx].second);
		Node* node = get_node(state, expand);
		if(node == nullptr){
			//playout結果を返す
			state.playout(reward, thread_id);
			return;
		}
		//UCTで着手を選択
		int action_idx = -1;
		bool expand_child = false;
		Action action = node->select(action_idx, expand_child, explore_coefficient, expansion_threshold, virtual_loss);
		assert(action_idx >= 0);
		//ロックを解除し, 子ノードへ
		lock.unlock();
		state.act(action);
		search_rec(state, reward, expand_child, thread_id);
		//プレイアウト結果を反映
		lock.lock();
		node->update(action_idx, reward, virtual_loss);
	}
}