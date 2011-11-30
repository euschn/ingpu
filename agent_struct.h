#pragma once

#include <thrust/version.h>
#include <iomanip>

#define MAX_PORTS 4

namespace ingpu {
	
enum EQUATION_TUPLE{LHS, RHS};

// Agents/Variables
struct Agent
{
	int		 id;		// unique id of the agent
	char	 name;		// name of the agent - determines agent (upper case) or variable (lower case)
	int		 arity;		// maximum arity = 4
	int		 ports[4];	// contains ids of connected agents

	//dummy constructor
	__host__ __device__
	Agent() : id(0), name('Z'), arity(0) {
		for (int i=0; i<arity; i++)
			ports[i] = 0;
	}	

	//copy ctor
	__host__ __device__
	Agent(const Agent& a) : id(a.id), name(a.name), arity(a.arity) {
		for (int i=0; i<arity; i++)
		ports[i] = a.ports[i];
	}
	
	__host__ __device__
	Agent(int _id, char _name, int _arity) : id(_id), name(_name), arity(_arity) {
		for (int i=0; i<arity; i++)
			ports[i] = 0;
	}
	
	__host__ __device__
	Agent(int _id, char _name, int _arity, int p0, int p1 = 0, int p2 = 0, int p3 = 0) : id(_id), name(_name), arity(_arity) {
			ports[0] = p0;
			ports[1] = p1;
			ports[2] = p2;
			ports[3] = p3;
	}

	__host__ __device__
	Agent(int _id) : id(_id), name('v'), arity(0) {
		for (int i=0; i<arity; i++)
			ports[i] = 0;
	}

	__host__ __device__
	bool operator==(const Agent& other)
	{
		if (name != other.name)
			return false;
		if (arity != other.arity)
			return false;
		if (id != other.id)
			return false;
		for (int i=0; i < arity; ++i) {
			if (ports[i] != other.ports[i])
				return false;
		}

		return true;
	}

	__host__ __device__
	bool operator!=(const Agent& other)
	{
		if (name != other.name)
			return true;
		if (arity != other.arity)
			return true;
		if (id != other.id)
			return true;
		for (int i=0; i < arity; ++i) {
			if (ports[i] != other.ports[i])
				return true;
		}

		return false;
	}

	__host__ __device__
	Agent& operator=( const Agent& a)
	{
		this->id = a.id;
		this->name = a.name;
		this->arity = a.arity;
		for (int i=0; i < a.arity; ++i) {
			this->ports[i] = a.ports[i];
		}
		return *this;
	}
};

}

// print an agent
__host__
std::ostream& operator<< (std::ostream& os, const ingpu::Agent& a);
