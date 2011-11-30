/*
	agent.h
	an agent struct and some functions
*/

#pragma once

#include "agent.h"
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <iomanip>

using namespace soa;
using namespace thrust;

namespace soa {
	
// our input for IN evaluation
typedef thrust::tuple<agent, agent, agent, agent, agent, agent, agent, agent, bool, int> input_tuple;
typedef thrust::unary_function<input_tuple, void> int_functor;

//typedefs for soa input
typedef device_vector<id_type>::iterator id_iter;
typedef device_vector<name_type>::iterator name_iter;
typedef device_vector<arity_type>::iterator arity_iter;
typedef device_vector<port_type>::iterator port_iter;
//tuple of these iterators
typedef tuple<id_iter, name_iter, arity_iter, port_iter, port_iter, port_iter, port_iter> agent_iter_tuple;
typedef zip_iterator<agent_iter_tuple> agent_iter;
typedef zip_iterator< tuple<agent_iter, agent_iter> > equation_iter;


__host__ __device__
agent dummy() 
{
	return thrust::make_tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type>(0, 'Z', 0, 0, 0, 0, 0);
}


__host__ __device__
agent make_agent(const id_type id, const name_type name, const arity_type arity)
{
	return thrust::make_tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type>(id, name, arity, 0, 0, 0, 0);
}
	
__host__ __device__
agent make_agent( const name_type name)
{
	return thrust::make_tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type>(0, name, 0, 0, 0, 0, 0);
}
	
__host__ __device__
agent make_agent( const id_type id)
{
	return thrust::make_tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type>(id, 'v', 0, 0, 0, 0, 0);
}

__host__ __device__
agent make_agent(const id_type id, const name_type name, const arity_type arity, const port_type p1, const port_type p2 = 0, const port_type p3 = 0, const port_type p4 = 0)
{
	return thrust::make_tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type>(id, name, arity, p1, p2, p3, p4);
}

__host__ __device__
bool is_upper(const name_type c)
{
	return (c >= 'A' && c <= 'Z');
}

__host__ __device__
bool is_lower(const name_type c)
{
	return (c >= 'a' && c <= 'z');
}

__host__ __device__
bool is_variable( const agent a)
{
	return is_lower(get<NAME>(a));
}
	
__host__ __device__
bool is_agent( const agent a)
{
	return is_upper(get<NAME>(a));
}

__host__ __device__
bool is_dummy( const agent a)
{
	return (get<ID>(a) == -1);
}

struct is_dummy_functor
{
	__host__ __device__
	bool operator() (const agent a)
	{
		return (get<ID>(a) == -1);
	}
};

struct is_dummy_equation_functor
{
	__host__ __device__
	bool operator() (const equation& e)
	{
		return (get<ID>(get<1>(e)) < 1) || (get<ID>(get<0>(e))  < 1);
	}
};

struct is_proper_equation_functor
{
	__host__ __device__
	bool operator() (const equation& e)
	{
		return (get<ID>(get<1>(e)) > 0) && (get<ID>(get<0>(e)) > 0) ;
	}
};

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

struct has_name_tuple : public thrust::unary_function<equation, bool>
{
	char name;

	__host__ __device__
	has_name_tuple(const char c) : name(c) {}
	
	__host__ __device__
	bool operator()(const equation& e) {
		return name == get<NAME>(get<1>(e));
	}
};

struct is_invalid_tuple : public thrust::unary_function<equation, bool>
{
	__host__ __device__
	bool operator()(const equation& e) {
		return (0 == get<NAME>(get<1>(e))) || (0 == get<NAME>(get<0>(e)));
	}
	//__host__ __device__
	//bool operator()(const equation& e) {
	//	return ((-1) > get<ID>(get<1>(e))) || ((-1) > get<ID>(get<0>(e)));
	//}
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


 //checks equations for top-level variables
 //returns 1 if lhs is a variable
	//	   2 if rhs is
	//	 3 for both
	//	 0 else
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


void print_equations(const thrust::host_vector<equation> equations)
{
	for( unsigned int i = 0; i < equations.size(); i++) {
		print_equation(get<LHS>(equations[i]), get<RHS>(equations[i]));
	}
	std::cout << std::endl;
}

void print_equations(const thrust::host_vector<soa::agent>& lhses, const thrust::host_vector<soa::agent>& rhses)
{
	for( unsigned int i = 0; i < lhses.size(); i++) {
		print_equation(lhses[i], rhses[i]);
	}
	std::cout << std::endl;
}


}
