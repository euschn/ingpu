
#pragma once

#include <thrust/version.h>
#include <thrust/tuple.h>
#include <iomanip>

using namespace thrust;

namespace soa {
	enum AGENT_TUPLE{ID, NAME, ARITY, P1, P2, P3, P4};
	enum EQUATION_TUPLE{LHS, RHS};
	// agent typedefs
	// TODO make name_type a string later on
	typedef char name_type;
	typedef int  id_type;
	typedef int  arity_type;
	typedef int  port_type;
	// an agent is basically a thrust tuple
	typedef tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type> agent;
	typedef tuple<agent, agent> equation;

}

// print an agent
__host__
std::ostream& operator<< (std::ostream& os, const soa::agent& a);

__host__
std::ostream& operator<< (std::ostream& os, const soa::equation& a);

void print_equation(const soa::agent& lhs, const soa::agent& rhs);