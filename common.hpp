#include <algorithm>
#include <vector>

namespace sheena{
	template<typename Ty, typename F>
	void remove(std::vector<Ty>& v, F f){
		v.erase(std::remove_if(v.begin(),v.end(), f), v.end());
	}
}