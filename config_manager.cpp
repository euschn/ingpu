
#include "config_manager.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

config_manager* config_manager::inst()
{
	static config_manager the_instance;
	return &the_instance;
}

void config_manager::load_config(const std::string& filepath)
{
	std::ifstream fp_in;
	fp_in.open(filepath, std::ios::in);
	if (!fp_in.is_open()) {
		std::cout << "error opening file " << filepath << std::endl;
		return;
	}

	string key;
	float value;

	while (getline(fp_in, key, ' '))
	{
		fp_in >> value;
		config_[key] = value;

		getline(fp_in, key);
	}
	std::cout << "done loading config\n" << endl;
}

const float config_manager::value(const std::string& key)
{
	config_map::iterator it;
	it = config_.find(key);
	if (it != config_.end()) {
		return it->second;
	}

	return -1;
}