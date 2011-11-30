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
#include "agent_struct.h"

using namespace thrust;

namespace ingpu {
	
// equation is a tuple of 2 agents
typedef thrust::tuple<Agent,Agent> equation;
// our input for IN evaluation
typedef thrust::tuple<Agent, Agent, Agent, Agent, Agent, Agent, Agent, Agent, bool, int> input_tuple;
typedef thrust::unary_function<input_tuple, void> int_functor;

// iterator types
typedef thrust::device_vector<Agent>::iterator agent_iter;
typedef thrust::tuple<agent_iter, agent_iter>  agent_iter_tuple;
typedef thrust::zip_iterator<agent_iter_tuple> equation_iter;


__host__ __device__
Agent get_lhs(equation eq)
{
	return thrust::get<0>(eq);
}

__host__ __device__
Agent get_rhs(equation eq)
{
	return thrust::get<1>(eq);
}


// device version of isupper(char)
__host__ __device__
bool is_upper(const char c)
{
	return (c >= 'A' && c <= 'Z');
}

__host__ __device__
bool is_lower(const char c)
{
	return (c >= 'a' && c <= 'z');
}

//agent is not a variable
__host__ __device__
bool is_agent(const Agent a)
{
	return is_upper(a.name);
}

//agent is a variable
__host__ __device__
bool is_variable(const Agent a)
{
	return is_lower(a.name);
}

struct has_name : public thrust::unary_function<Agent, bool>
{
	char name;

	__host__ __device__
	has_name(const char c) : name(c) {}
	
	__host__ __device__
	bool operator()(const Agent& agent) {
		return name == agent.name;
	}
};

//debug - look for specific symbols
struct has_name_special : public thrust::unary_function<Agent, bool>
{
	
	__host__ __device__
	bool operator()(const Agent& agent) {
		//return 'S' == agent.name || 'N' == agent.name || 'M' == agent.name;
		return 'A' != agent.name && 'B' != agent.name;
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
		// if both LHSes are variables, compare the RHSes
		// agents (upper case) precede variables (lower case)
		char c = get<0>(first).name;
		char d = get<0>(second).name;
		if ( (c >= 'a' && c <= 'z') && (d >= 'a' && d <= 'z') ) {
			return (get<0>(first).id < get<0>(second).id);
		}

		//default: compare lhs names lexicographically
		return (get<0>(first).name > get<0>(second).name);
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
	if (is_variable(get_lhs(e)))
		result++;
	if (is_variable(get_rhs(e)))
		result += 2;

	return result;
}

// checks for tlv, boolean version
__host__ __device__
bool has_toplevel_variable(const equation& e)
{
	if (is_variable(get_lhs(e)))
		return true;
	else 
		return is_variable(get_rhs(e));
}

//orient equations t=x -> x=t
struct orient_functor
{
	template <typename Equ>
	__host__ __device__
	void operator()(Equ eq)
	{
		if (!is_variable(get_lhs(eq)) && is_variable(get_rhs(eq)))  
		{
			thrust::swap(thrust::get<0>(eq), thrust::get<1>(eq));
		}
	}
};


//is an equation active
struct is_active_functor
{
	template <typename Equ>
	__host__ __device__
	bool operator()(Equ eq)
	{
		const char c = thrust::get<0>(eq).name;
		return (c >= 'A' && c <= 'Z');
	}
};

//is an equation active
struct is_not_active_functor
{
	template <typename Equ>
	__host__ __device__
	bool operator()(Equ eq)
	{
		const char c = thrust::get<0>(eq).name;
		return !(c >= 'A' && c <= 'Z');
	}
};


//is an equation active
struct is_active_agent
{
	__host__ __device__
	bool operator()(const Agent& a)
	{
		const char c = a.name;
		return (c >= 'A' && c <= 'Z');
	}
};

//orient equations t=x -> x=t
struct orient_functor_transform : public thrust::unary_function<equation, equation>
{
	__host__ __device__
	equation operator()(const equation eq)
	{
		equation result(eq);
		if (!is_variable(get_lhs(eq)) && is_variable(get_rhs(eq)))  
		{
			thrust::swap(thrust::get<0>(result), thrust::get<1>(result));
		}

		return result;
	}
};

void print_equations(const thrust::host_vector<Agent>& lhses, const thrust::host_vector<Agent>& rhses)
{
		for(unsigned int i = 0; i < lhses.size(); i++) {
			std::string left_args, right_args;
			if (lhses[i].arity == 0) {
				left_args = "";
			}
			else {
				
				std::stringstream out;
				out << "(" << lhses[i].ports[0];
				for (unsigned int ar = 1; ar < lhses[i].arity; ar++) {
					out << ", ";
					out << lhses[i].ports[ar];
				}
				out << ")";

				left_args = out.str();
			}

			if (rhses[i].arity == 0) {
				right_args = "   ";
			}
			else {
				std::stringstream out;
				out << "(" << rhses[i].ports[0];
				for (int ar = 1; ar < rhses[i].arity; ar++) {
					out << ", ";
					out << rhses[i].ports[ar];
				}
				out << ")";

				right_args = out.str();
			}
			std::cout << lhses[i]
				<< left_args 
				<< " = "
				<< rhses[i] 
				<< right_args 
				<< std::setw(10) <<  "  (" 
				<< lhses[i].id << ", " << rhses[i].id  << ") \n";
		}
		std::cout << std::endl;
}

void print_equations(const thrust::host_vector<equation> equations)
{
		for(unsigned int i = 0; i < equations.size(); i++) {
			std::string left_args, right_args;
			if (get_lhs(equations[i]).arity == 0) {
				left_args = "";
			}
			else {
				
				std::stringstream out;
				out << "(" << get_lhs(equations[i]).ports[0];
				for (unsigned int ar = 1; ar < get_lhs(equations[i]).arity; ar++) {
					out << ", ";
					out << get_lhs(equations[i]).ports[ar];
				}
				out << ")";

				left_args = out.str();
			}

			if (get_rhs(equations[i]).arity == 0) {
				right_args = "   ";
			}
			else {
				std::stringstream out;
				out << "(" << get_rhs(equations[i]).ports[0];
				for (int ar = 1; ar < get_rhs(equations[i]).arity; ar++) {
					out << ", ";
					out << get_rhs(equations[i]).ports[ar];
				}
				out << ")";

				right_args = out.str();
			}
			std::cout << get_lhs(equations[i])
				<< left_args 
				<< " = "
				<< get_rhs(equations[i])
				<< right_args 
				<< std::setw(10) << "  (" 
				<< get_lhs(equations[i]).id << ", " << get_rhs(equations[i]).id  << ") \n";
		}
		std::cout << std::endl;
}



void print_equations(const equation_iter begin, const equation_iter end)
{
	thrust::host_vector<equation> equations(begin, end);
	print_equations(equations);
}


}


#endif