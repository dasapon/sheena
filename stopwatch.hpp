#include <chrono>
namespace sheena{
	class Stopwatch {
		std::chrono::system_clock::time_point start;
	public:
		Stopwatch() { restart(); }
		void restart(){ start = std::chrono::system_clock::now(); }
		uint64_t msec()const {
			using namespace std::chrono;
			return duration_cast<milliseconds>(system_clock::now() - start).count();
		}
		uint64_t sec()const{
			using namespace std::chrono;
			return duration_cast<seconds>(system_clock::now() - start).count();
		}
		uint64_t min()const{
			using namespace std::chrono;
			return duration_cast<minutes>(system_clock::now() - start).count();
		}
	};
}