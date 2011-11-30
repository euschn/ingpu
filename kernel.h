#pragma once

#include <thrust\host_vector.h>
#include "agent_struct.h"
#include "result_info.h"

result_info test_interaction_step(const int ruleset, thrust::host_vector<ingpu::Agent>& in_lhs_host, thrust::host_vector<ingpu::Agent>& in_rhs_host, const bool verbose, const bool print_final_result=true,
	const bool pause=false);

result_info partition_interaction_step(const int ruleset, thrust::host_vector<ingpu::Agent>& in_lhs_host, thrust::host_vector<ingpu::Agent>& in_rhs_host, const bool verbose, const bool print_final_result=true,
	const bool pause=false);