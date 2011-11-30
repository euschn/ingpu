/*
 communication.h
 does a communication step
*/

#ifndef COMMUNICATION_H_
#define COMMUNICATION_H_

//#pragma once

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include "agent.cuh"
#include <stopwatch.h>
#include <stopwatch_win.h>

using namespace ingpu;

namespace ingpu {

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
			
				if (get<LHS>(eq).id == tlv_id) {
					return true;
				}
				else {
					//check if the rhs is a variable
					//Agent r = get_rhs(t);
					if (is_variable(get_rhs(eq))) {

						return (get_rhs(eq).id == tlv_id); //neither agent is a variable
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
			
				if (get<LHS>(eq).id == tlv_id) {
					return true;
				}
				else {
					//check if the rhs is a variable
					//Agent r = get_rhs(t);
					if (!is_agent(get_rhs(eq))) {

						return (get_rhs(eq).id == tlv_id); //neither agent is a variable
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
				return (!is_agent(get_rhs(t)));
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
			return is_variable(get<LHS>(t)) && is_agent(get_rhs(t));
		}
	};

	// dummy equation test functor
	struct dummy_test_functor
	{
		__host__ __device__
		bool operator()(const Agent t)
		{
			return (t.name == 'Z');
		}
	};

	struct dummy_test_functor_tuple
	{
		__host__ __device__
		bool operator()(const thrust::tuple<Agent, Agent>& t)
		{
			return (thrust::get<0>(t).name == 'Z');
		}
	};

	struct valid_tuple_functor
	{
		__host__ __device__
		bool operator()(const thrust::tuple<Agent, Agent>& t)
		{
			return (thrust::get<0>(t).name != 'Z');
		}
	};

	// do two equations have a common tlv?
	// assumes equations are oriented
	struct common_tlv_functor : public thrust::binary_function<equation, equation, bool>
	{
		__host__ __device__
		bool operator()(equation t, equation s)
		{
			//TODO: we assume that no two non-variable agents have distinct ids
			// this may cause bugs when ids are assigned to non-variable agents more than once
			// (which should not happen)

			if (get<LHS>(t).id == get<LHS>(s).id)
				return true;
			
			else if (get<LHS>(t).id == get<RHS>(s).id)
				return true;

			else if (get<RHS>(t).id == get<LHS>(s).id)
				return true;

			else if (get<RHS>(t).id == get<RHS>(s).id)
				return true;

			return false;
		}
	};

	struct smaller_tlv_functor : public thrust::binary_function<equation, equation, bool>
	{
		__host__ __device__
		bool operator()(const equation& t, const equation& s)
		{
			//TODO: consider LHS tlvs only

			return get<0>(t).id < get<0>(s).id;
		}
	};
	

	struct greater_tlv_functor : public thrust::binary_function<equation, equation, bool>
	{
		__host__ __device__
		bool operator()(equation t, equation s)
		{
			//TODO: consider LHS tlvs only

			return get<LHS>(t).id > get<LHS>(s).id;
		}
	};

	struct equal_tlv_functor : public thrust::binary_function<equation, equation, bool>
	{
		__host__ __device__
		bool operator()(equation t, equation s)
		{
			//TODO: consider LHS tlvs only

			return get<LHS>(t).id == get<LHS>(s).id;
		}
	};

	// retuns the tlv id of the LHS
	struct get_tlv_functor : public thrust::unary_function<equation, int>
	{
		__host__ __device__
		int operator()(const equation& t)
		{
			//TODO: consider LHS tlvs only
			return get<0>(t).id;
		}
	};
	
	// retuns the tlv id of the LHS
	struct get_tlv_single_functor : public thrust::unary_function<Agent, int>
	{
		__host__ __device__
		int operator()(const Agent& t)
		{
			//TODO: consider LHS tlvs only
			return t.id;
		}
	};

	struct different_tlv_functor : public thrust::binary_function<equation, equation, bool>
	{
		__host__ __device__
		bool operator()(equation t, equation s)
		{
			//TODO: consider LHS tlvs only

			return !(get<LHS>(t).id == get<LHS>(s).id);
		}
	};

	struct comm_functor : public thrust::binary_function<equation, equation, equation>
	{
		__host__ __device__
		equation operator()(equation t, equation s)
		{
			//TODO: consider LHS tlvs only

			const char c = get<RHS>(s).id;
			// swap lhs/rhs if rhs is a tlv
			if  (c >= 'A' && c <= 'Z') {
				return thrust::make_tuple<Agent, Agent>(get<RHS>(s), get<RHS>(t));
			}
			//TODO: more efficient to add 'else' here?
			return thrust::make_tuple<Agent, Agent>(get<RHS>(t), get<RHS>(s));
		}
	};

	// performs a single communication
	// by resolving a common tlv
	struct comm_reduce_functor : public thrust::binary_function<equation, equation, equation>
	{
		__host__ __device__
		equation operator()(const equation& t, const equation& s)
		{
			if (get<LHS>(t).id == get<LHS>(s).id) {
				//orient if necessary
				//make the agent with the smaller name the LHS
				//this means that if one of them is a tlv and the other an agent,
				//the tlv will be on the lhs
				if (get<RHS>(t).name < get<RHS>(s).name)  
				{
					return thrust::make_tuple<Agent, Agent>(get<RHS>(s), get<RHS>(t));//thrust::swap(get<RHS>(t), get<RHS>(s));
				}
				return thrust::make_tuple<Agent, Agent>(get<RHS>(t), get<RHS>(s));
			}

			else if (get<LHS>(t).id == get<RHS>(s).id) {
				//orient if necessary
				if (get<RHS>(t).name < get<LHS>(s).name)  
				{
					return thrust::make_tuple<Agent, Agent>( get<LHS>(s), get<RHS>(t));//thrust::swap(get<RHS>(t), get<LHS>(s));
				}
				return thrust::make_tuple<Agent, Agent>(get<RHS>(t), get<LHS>(s));
			}

			else if (get<RHS>(t).id == get<LHS>(s).id) {
				//orient if necessary
				if (get<LHS>(t).name < get<RHS>(s).name)  
				{
					return thrust::make_tuple<Agent, Agent>(get<RHS>(s), get<LHS>(t));//thrust::swap(get<LHS>(t), get<RHS>(s));
				}
				return thrust::make_tuple<Agent, Agent>(get<LHS>(t), get<RHS>(s));
			}

			else if (get<RHS>(t).id == get<RHS>(s).id) {
				//orient if necessary
				if (get<LHS>(t).name < get<LHS>(s).name)  
				{
					return thrust::make_tuple<Agent, Agent>(get<LHS>(s), get<LHS>(t));//thrust::swap(get<LHS>(t), get<LHS>(s));
				}
				return thrust::make_tuple<Agent, Agent>(get<LHS>(t), get<LHS>(s));
			}

			// at this point, something has gone wrong
			// (no common tlv)
			// return an equation of dummy agents
			return thrust::make_tuple<Agent, Agent>(Agent(), Agent());
		}
	};

	
	//clear the additional result arrays
	struct clear_functor {

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple& equations)
		{
			thrust::get<0>(equations).name = 'Z';
			thrust::get<1>(equations).name = 'Z';
			thrust::get<2>(equations).name = 'Z';
			thrust::get<3>(equations).name = 'Z';
			thrust::get<4>(equations).name = 'Z';
			thrust::get<5>(equations).name = 'Z';
			thrust::get<6>(equations) = false;
		}

	};

	// parallel communication step using reduce_by_key
	int comm_step_parallel(equation_iter& start, equation_iter& end, equation_iter& result_start, equation_iter& result_end, const int input_size, const int loop, const unsigned int communication_watch = 0)
	{
		int comm_counter = 1;
		//equation_iter new_end;

		//StopWatch::get(communication_watch).start();
		
		thrust::device_vector<int> keys(input_size);
		thrust::device_vector<int> keys_out(input_size);
		thrust::transform(start, end, keys.begin(), get_tlv_functor());

		//thrust::adjacent_difference(ids_begin, ids_end, diffs.begin());

		//new_end = thrust::reduce_by_key(start, end, start, 
		//	thrust::make_discard_iterator(), result_start, common_tlv_functor(), comm_reduce_functor()
		//	).second;

		auto new_end = thrust::reduce_by_key(keys.begin(), keys.end(), start, 
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

#endif