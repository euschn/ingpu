#pragma once

#include <thrust\host_vector.h>
//#include "agent.h"
#include "../result_info.h"


using namespace thrust;

namespace soa {

result_info interaction_loop(
	const int ruleset, 
	host_vector<int>& host_lhs_ids,
	host_vector<char>& host_lhs_names,
	host_vector<int>& host_lhs_arities,
	host_vector<int>& host_lhs_p1s,
	host_vector<int>& host_lhs_p2s,
	host_vector<int>& host_lhs_p3s,
	host_vector<int>& host_lhs_p4s,
	host_vector<int>& host_rhs_ids,
	host_vector<char>& host_rhs_names,
	host_vector<int>& host_rhs_arities,
	host_vector<int>& host_rhs_p1s,
	host_vector<int>& host_rhs_p2s,
	host_vector<int>& host_rhs_p3s,
	host_vector<int>& host_rhs_p4s,
	const bool verbose, const bool print_final, const bool pause);

}