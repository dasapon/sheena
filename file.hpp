#pragma once

#include <fstream>
#include "common.hpp"

namespace sheena{
	inline bool read_file(std::string file_name, std::vector<std::string>& lines){
		std::ifstream file(file_name);
		if(!file.is_open())return false;
		std::string line;
		while(std::getline(file, line)){
			lines.push_back(line);
		}
		file.close();
		return true;
	}
}