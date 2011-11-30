//info class about a result of an interaction

#pragma once

#include <vector>

class result_info {
public:
	result_info();
	result_info(const int result, 
		const int num_loops, 
		const int total_interactions, 
		const float com_time=0, 
		const float int_time=0);

	//members
	int result;
	int num_loops;
	int total_interactions;
	float com_time;
	float int_time;
	std::vector<int> loops_per_step;


};