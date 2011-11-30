/*
 communication.h
 does a communication step
 SOA version
*/

#pragma once

//#pragma once

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <stopwatch.h>
#include <stopwatch_win.h>
#include "agent_functor.h"

using namespace soa;

namespace soa {

// do two agents have the same top level variable (TLV)
struct has_tlv_functor : public thrust::unary_function<equation,bool>
	{
		int tlv_id;

		__host__ __device__
		has_tlv_functor(const int _tlv_id) : tlv_id(_tlv_id) {}
	
		__host__ __device__
		bool operator()(const equation& eq) const
		{
			//check if the lhs is a variable
			if ( is_variable( get<LHS>(eq) ) ) {
			
				if (get<ID>(get<LHS>(eq)) == tlv_id) {
					return true;
				}
				else {
					//check if the rhs is a variable
					//agent r = get<RHS>(t);
					if (is_variable(get<RHS>(eq))) {

						return (get<ID>(get<RHS>(eq)) == tlv_id); //neither agent is a variable
					}
				}
			}
			return false;
		}
};

// do two agents have the same top level variable (TLV)
struct find_tlv_functor : public thrust::unary_function<equation,bool>
	{
		int tlv_id;

		__host__ __device__
		find_tlv_functor(const int _tlv_id) : tlv_id(tlv_id) {}
	
		__host__ __device__
		bool operator()(const equation& eq) const
		{
			//check if the lhs is a variable
			if (!is_agent(get<LHS>(eq))) {
			
				if (get<ID>( get<LHS>(eq) ) == tlv_id) {
					return true;
				}
				else {
					//check if the rhs is a variable
					//agent r = get<RHS>(t);
					if (!is_agent(get<RHS>(eq))) {

						return (get<ID>( get<RHS>(eq) ) == tlv_id); //neither agent is a variable
					}
				}
			}
			return false;
		}
};




	// does an equation have a TLV
	struct is_tlv_equation_functor
	{
		__host__ __device__
		bool operator()(equation t)
		{
			//check if the lhs is a variable
			if (!is_agent(get<LHS>(t))) {
					return true;
			}
			else {
				//check if the rhs is a variable
				return (!is_agent(get<RHS>(t)));
			}
		}
	};

	// does an equation have a TLV as its lhs?
	struct is_oriented_tlv_equation_functor
	{
		__host__ __device__
		bool operator()(equation t)
		{
			//check if the lhs is a variable
			return is_variable(get<LHS>(t)) && is_agent(get<RHS>(t));
		}
	};

	// TODO dont check name, check ID ( == -1)
	// dummy equation test functor
	struct dummy_test_functor
	{
		__host__ __device__
		bool operator()(const agent t)
		{
			return (get<NAME>(t) == 'Z');
		}
	};

	struct dummy_test_functor_tuple
	{
		__host__ __device__
		bool operator()(const thrust::tuple<agent, agent> t)
		{
			agent l = thrust::get<0>(t);
			return (get<NAME>(l) == 'Z');
		}
	};

	// do two equations have a common tlv?
	// assumes equations are oriented
	struct common_tlv_functor : public thrust::binary_function<equation, equation, bool>
	{
		__host__ __device__
		bool operator()(const equation& t, const equation& s)
		{
			//TODO: we assume that no two non-variable agents have distinct ids
			// this may cause bugs when ids are assigned to non-variable agents more than once
			// (which should not happen)
			agent t_lhs = get<LHS>(t);
			agent s_lhs = get<LHS>(s);
			if (get<ID>(t_lhs) == get<ID>(s_lhs) )
				return true;
			
			agent s_rhs = get<RHS>(s);
			if (get<ID>(t_lhs) == get<ID>(s_rhs))
				return true;

			agent t_rhs = get<RHS>(t);
			if (get<ID>(t_rhs) == get<ID>(s_lhs))
				return true;

			if (get<ID>(t_rhs) == get<ID>(s_rhs) )
				return true;

			return false;
		}
	};

	// performs a single communication
	// by resolving a common tlv
	struct comm_reduce_functor : public thrust::binary_function<equation, equation, equation>
	{
		__host__ __device__
		equation operator()(const equation& t, const equation& s)
		{
		
			agent t_lhs = get<LHS>(t);
			agent t_rhs = get<RHS>(t);
			agent s_lhs = get<LHS>(s);
			agent s_rhs = get<RHS>(s);

			if (get<ID>(t_lhs) == get<ID>(s_lhs) ) {
				//orient if necessary
				if (!is_variable(t_rhs) && is_variable(s_rhs))  
				{
					thrust::swap(t_rhs, s_rhs);
				}
				return thrust::make_tuple<agent, agent>(t_rhs, s_rhs);
			}

			if (get<ID>(t_lhs) == get<ID>(s_rhs)) {
				//orient if necessary
				if (!is_variable(t_rhs) && is_variable(s_lhs))  
				{
					thrust::swap(t_rhs, s_lhs);
				}
				return thrust::make_tuple<agent, agent>(t_rhs, s_lhs);
			}

			if (get<ID>(t_rhs) == get<ID>(s_lhs)) {
				//orient if necessary
				if (!is_variable(t_lhs) && is_variable(s_rhs))  
				{
					thrust::swap(t_lhs, s_rhs);
				}
				return thrust::make_tuple<agent, agent>(t_lhs, s_rhs);
			}

			if (get<ID>(t_rhs) == get<ID>(s_rhs)) {
				//orient if necessary
				if (!is_variable(t_lhs) && is_variable(s_lhs))  
				{
					thrust::swap(t_lhs, s_lhs);
				}
				return thrust::make_tuple<agent, agent>(t_lhs, s_lhs);
			}

			// at this point, something has gone wrong
			// (no common tlv)
			// return an equation of dummy agents
			return thrust::make_tuple<agent, agent>(agent(), agent());
		}
	};

	// parallel communication step using reduce_by_key
	int comm_step_parallel(equation_iter& start, equation_iter& end, 
		equation_iter& result_start, equation_iter& result_end, 
		device_vector<int>::iterator& keys_start, device_vector<int>::iterator& keys_end,
		const int input_size, const unsigned int communication_watch = 0)
	{
		int comm_counter = 1;

		//StopWatch::get(communication_watch).start();
		
		thrust::device_vector<int> keys_out(input_size);

		//TODO improve comm_reduce functor
		auto new_end = thrust::reduce_by_key(keys_start, keys_end, start, 
			keys_out.begin(), result_start, thrust::equal_to<int>(), comm_reduce_functor()
			);

		//StopWatch::get(communication_watch).stop();

		int diff = new_end.first - keys_out.begin();
		diff = input_size - diff;
		//std::cout << diff << std::endl;
		if (diff == 0) {
			return 0;
		}
		result_end = new_end.second;
		//end = end - diff;
		
		return diff;
	}


}