.PHONY:all test
all:test benchmark
test_executable:
	clang++ -std=c++17 -Wall -march=native -o test/bin/test_executable test/main.cpp -lpthread
bench_executable:
	clang++ -std=c++17 -Wall -march=native -O3 -DNDEBUG -o test/bin/bench_executable test/benchmark.cpp -lpthread
test:test_executable 
	test/bin/test_executable
benchmark:bench_executable
	test/bin/bench_executable