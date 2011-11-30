// class config manager
// singleton

#pragma once

#include <map>

typedef std::map<std::string, float> config_map;

class config_manager
{
public:
	//get the instance
	static config_manager* inst();

	static void load(const std::string& filepath) {inst()->load_config(filepath); }
	static float get(const std::string& key) { return inst()->value(key); }
	static int get_int(const std::string& key) { return static_cast<int>(inst()->value(key)); }
	static bool get_bool(const std::string& key) { if (inst()->value(key) == 0) return false; else return true; }

	void load_config(const std::string& filepath);
	const config_map config() const {return config_;}
	const float value(const std::string& key); 

private:
	config_manager() {}
	virtual ~config_manager() {}


	config_map config_;

};