.PHONY:all test
EXE=executable
all:test
executable:
	clang++ -std=c++17 -Wall -march=native -o test/bin/$(EXE) test/main.cpp -lpthread
test:executable 
	test/bin/$(EXE)