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
	template<char delim>
	bool read_separated_values(std::string file_name, std::vector<std::vector<std::string>>& lines){
		std::vector<std::string> buffer;
		if(!read_file(file_name, buffer))return false;
		for(const auto& str : buffer){
			lines.push_back(split_string(str, delim));
		}
		return true;
	}
}