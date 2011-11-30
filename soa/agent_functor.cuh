/*
	agent.h
	an agent struct and some functions
*/

//#pragma once
#ifndef AGENT_H_
#define AGENT_H_

//#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "agent.h"
#include <iomanip>

using namespace soa;

namespace soa {
	
// our input for IN evaluation
typedef thrust::tuple<agent, agent, agent, agent, agent, agent, agent, agent, bool, int> input_tuple;
typedef thrust::unary_function<input_tuple, void> int_functor;

// iterator types
typedef thrust::device_vector<agent>::iterator agent_iter;
typedef thrust::tuple<agent_iter, agent_iter>  agent_iter_tuple;
typedef thrust::zip_iterator<agent_iter_tuple> equation_iter;

struct has_name : public thrust::unary_function<agent, bool>
{
	char name;

	__host__ __device__
	has_name(const char c) : name(c) {}
	
	__host__ __device__
	bool operator()(const agent& agent) {
		return name == get<NAME>(agent);
	}
};

// greater comparison function
// oriented communication equations are smaller than everything
// else, compare lexicographically
struct comm_equation_greater : public thrust::binary_function<equation, equation, bool>
{

	__host__ __device__
	bool operator()(const equation& first, const equation& second)
	{
		//Agent lhs_first = get<LHS>(first);
		//Agent rhs_first = get<RHS>(first);
		//
		//Agent lhs_second = get<LHS>(second);
		//Agent rhs_second = get<RHS>(second);

		// if both LHSes are variables, compare the RHSes
		// agents (upper case) precede variables (lower case)
		if ( is_variable(get<LHS>(first) ) && is_variable(get<LHS>(second)) ) {
			return (get<ID>(get<LHS>(first)) < get<ID>(get<LHS>(second)) );
		}

		//default: compare lhs names lexicographically
		return ( get<NAME>(get<LHS>(first)) > get<NAME>(get<LHS>(second)));
	}
};


// checks equations for top-level variables
// returns 1 if lhs is a variable
//		   2 if rhs is
//         3 for both
//         0 else
__host__ __device__
int top_level_variables(const equation& e)
{
	int result = 0;
	if (is_variable(get<LHS>(e)))
		result++;
	if (is_variable(get<RHS>(e)))
		result += 2;

	return result;
}

// checks for tlv, boolean version
__host__ __device__
bool has_toplevel_variable(const equation& e)
{
	if (is_variable(get<LHS>(e)))
		return true;
	else 
		return is_variable(get<RHS>(e));
}

//orient equations t=x -> x=t
struct orient_functor
{
	template <typename Equ>
	__host__ __device__
	void operator()(Equ eq)
	{
		if (!is_variable(get<LHS>(eq)) && is_variable(get<RHS>(eq)))  
		{
			thrust::swap(get<LHS>(eq), get<RHS>(eq));
		}
	}
};

//orient equations t=x -> x=t
struct orient_functor_transform : public thrust::unary_function<equation, equation>
{
	__host__ __device__
	equation operator()(const equation eq)
	{
		equation result(eq);
		if (!is_variable(get<LHS>(eq)) && is_variable(get<RHS>(eq)))  
		{
			thrust::swap(get<LHS>(result), get<RHS>(result));
		}

		return result;
	}
};


__host__
void print_equations(const thrust::host_vector<agent>& lhses, const thrust::host_vector<agent>& rhses)
{
	for( unsigned int i = 0; i < lhses.size(); i++) {
		print_equation(lhses[i], rhses[i]);
	}
	std::cout << std::endl;
}

void print_equations(const thrust::host_vector<equation> equations)
{
	for( unsigned int i = 0; i < equations.size(); i++) {
		print_equation(get<LHS>(equations[i]), get<RHS>(equations[i]));
	}
	std::cout << std::endl;
}

}


#endif