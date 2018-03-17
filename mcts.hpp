
#include <cmath>

#include "common.hpp"

namespace sheena::mcts{
	//汎用のMCTSを実現する
	//Stateに実装される必要のあるメソッド
	//コピーコンストラクタ
	//State(const State&)
	//actionを実行し,状態遷移する
	//void act(Action)
	//playoutを行い, 報酬を得る
	//void playout(sheena::Array<double, NPlayer>&)
	//そのStateにおける行動とそれらの事前確率を得る
	//戻り値は手番プレイヤの番号(0, 1, ... NPlayer-1)
	//また、ゲーム終了時には可能な行動数を0として返す
	//int get_actions(int&, sheena::Array<Action, MaxAction>&, sheena::Array<float, MaxAction>&)const;
	//局面のハッシュ値を返す
	//uint64_t key()const;
	template<typename State, typename Action, size_t NPlayer, size_t MaxAction>
	class Searcher{
		static constexpr int chain_size = 4;
		double explore_coefficient;
		int expansion_threshold;
		struct Edge{
			double reward;
			uint64_t played_plus_one;
			Edge():reward(0), played_plus_one(1){}
			void update(double r){
				reward = (reward * (played_plus_one - 1) + r) / (played_plus_one);
				played_plus_one++;
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
			void update(int act_idx, Array<double, NPlayer>& reward){
				edges[act_idx].update(reward[turn_player]);
			}
			Action puct(int& idx, bool& expand, double C, int exp_th){
				double max_ucb = -DBL_MAX;
				double expl = std::sqrt(double(total_played)) * C;
				total_played++;
				for(int i=0;i<n_action;i++){
					double ucb = edges[i].reward + prior_probability[i] * expl / edges[i].played_plus_one;
					if(max_ucb < ucb){
						idx = i;
						max_ucb = ucb;
					}
				}
				expand = edges[idx].played_plus_one > exp_th + 1;
				return actions[idx];
			}
			int search_result(Array<Action, MaxAction>& actions, Array<double, MaxAction>& rewards, 
			Array<int, MaxAction>& count)const{
				int total = 0;
				for(int i=0;i<n_action;i++){
					actions[i] = this->actions[i];
					rewards[i] = this->edges[i].reward;
					count[i] = this->edges[i].played_plus_one - 1;
					total += count[i];
				}
				assert(total == total_played);
				return n_action;
			}
		};
		ArrayAlloc<Array<Node, chain_size>> tt;
		int generation_;
		Node* get_node(State& state, bool expand){
			uint64_t key = state.key();
			size_t idx = key % tt.size();
			int empty = -1;
			//todo ロックする
			for(int i=0;i<chain_size;i++){
				if(tt[idx][i].key() == key){
					tt[idx][i].update_generation(generation_);
					return &tt[idx][i];
				}
				if(tt[idx][i].generation() != generation_)empty = i;
			}
			//開いている箇所を使ってノード展開
			if(empty != -1 && expand){
				if(tt[idx][empty].set_up(state, generation_))return &tt[idx][empty];
			}
			return nullptr;
		}
		void search_rec(State& state, Array<double, NPlayer>& reward, bool expand);
	public:
		Searcher():explore_coefficient(1.0), expansion_threshold(0), tt(16384){
			generation_ = 1;
			clear_tt();
		}
		void search(const State& root, size_t po){
			generation_++;
			if(generation_ < 0)generation_ = 1;
			sheena::Array<double, NPlayer> reward;
			for(int i=0;i<po;i++){
				State state(root);
				search_rec(state, reward, true);
			}
		}
		void clear_tt(){
			for(size_t i=0;i<tt.size();i++){
				for(int j=0;j<chain_size;j++)tt[i][j].invalidate(i);
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
		//stateにおける各行動の過去の報酬の平均と、play回数を得る
		//戻り値は合法手数(そもそもそのstateがtree中に無い場合は-1)
		int search_result(const State& state, Array<Action, MaxAction>& actions, 
		Array<double, MaxAction>& rewards, Array<int, MaxAction>& count)const{
			uint64_t key = state.key();
			size_t idx = key % tt.size();
			for(int i=0;i<chain_size;i++){
				if(tt[idx][i].key() == key){
					return tt[idx][i].search_result(actions, rewards, count);
				}
			}
			return -1;
		}
	};
	template<typename State, typename Action, size_t NPlayer, size_t MaxAction>
	void Searcher<State, Action, NPlayer, MaxAction>::search_rec(State& state, Array<double, NPlayer>& reward, bool expand){
		//Nodeを取得し,(todo ロックを取る)
		Node* node = get_node(state, expand);
		if(node == nullptr){
			//playout結果を返す
			state.playout(reward);
			return;
		}
		//UCTで着手を選択
		int action_idx = -1;
		bool expand_child = false;
		Action action = node->puct(action_idx, expand_child, explore_coefficient, expansion_threshold);
		assert(action_idx >= 0);
		//(todo ロックを解除し), 子ノードへ
		state.act(action);
		search_rec(state, reward, expand_child);
		//プレイアウト結果を反映
		node->update(action_idx, reward);
	}
}