
#include "result_info.h"


result_info::result_info()
{
	result=0;
	num_loops=1;
	total_interactions=0;
	loops_per_step = std::vector<int>(0);
}

result_info::result_info(const int result, const int num_loops, const int total_interactions, 
		const float com_time, 
		const float int_time)
{
	this->result = result;
	this->num_loops = num_loops;
	this->total_interactions = total_interactions;
	this->com_time = com_time;
	this->int_time = int_time;
	loops_per_step = std::vector<int>(0);
}