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
		static constexpr double explore_coefficient = 1.0;
		static constexpr int chain_size = 4;
		static constexpr int expand_threshold = 2;
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
			int n_moves;
			int total_played;
			sheena::Array<Action, MaxAction> actions;
			sheena::Array<float, MaxAction> prior_probability;
			sheena::Array<Edge, MaxAction> edges;
		public:
			Node():generation_(-1), n_moves(0){}
			void invalidate(uint64_t idx){
				generation_ = -1;
				//keyが合わないようにする
				key_ = idx + 1;
			}
			uint64_t key()const{return key_;}
			int generation()const{return generation_;}
			bool set_up(const State& state, int gen){
				generation_ = gen;
				key_ = state.key();
				total_played = 0;
				turn_player = state.get_actions(n_moves, actions, prior_probability);
				if(n_moves == 0){
					//終端ノードは置換表には不要
					invalidate(key_);
					return false;
				}
				return true;
			}
			void update(int act_idx, Array<double, NPlayer>& reward){
				edges[act_idx].update(reward[turn_player]);
			}
			Action select_action(int& idx, bool& expand){
				double max_ucb = -DBL_MAX;
				double expl = std::sqrt<double>(total_played) * explore_coefficient;
				for(int i=0;i<n_moves;i++){
					double ucb = edges[i].reward + prior_probability[i] * expl / edges[i].played_plus_one;
					if(max_ucb < ucb)idx = i;
				}
				expand = edges[idx].playout_plus_one > expand_threshold;
				return actions[idx];
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
				if(tt[idx][i].ok())return &tt[idx][i];
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
		Searcher():tt(16384){
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
			tt.resize(sz / 2);
			clear_tt();
		}
	};
	template<typename State, typename Action, size_t NPlayer, size_t MaxAction>
	void Searcher<State, Action, NPlayer, MaxAction>::search_rec(State& state, Array<double, NPlayer>& reward, bool expand){
		//Nodeを取得し,ロックを取る(並列化する場合)
		Node* node = get_node(state, expand);
		if(node == nullptr){
			//playout結果を返す
			state.playout(reward);
			return;
		}
		//UCTで着手を選択
		int action_idx = -1;
		bool expand_child = false;
		Action action = node->select_action(action_idx, expand_child);
		assert(action_idx >= 0);
		//ロックを解除し, 子ノードへ
		state.act(action);
		search_rec(state, reward, expand_child);
		//プレイアウト結果を反映
		node->update(action_idx, reward);
	}
}