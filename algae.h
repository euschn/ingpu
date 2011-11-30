// algae functor
// computes nth iteration of original L-system to model algae groth
// en.wikipedia.org/wiki/L-system

//IMPORTANT
// RHS equations must be oriented

//#pragma once
#ifndef ALGAE_H_
#define ALGAE_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <cutil_inline.h>
#include <cutil.h>
//#include "device_functions.h"

#include "agent.cuh"

using namespace ingpu;
using namespace thrust;

//max new indices per interaction
const int algae_max_idx = 9;

struct algae_functor : public int_functor {

int * index_ptr;
__host__ __device__
algae_functor(int * ptr) : index_ptr(ptr) {}


template <typename Tuple>
__device__
void operator()(Tuple equations)
{
	
	//get lhs and rhs of the equations
	Agent lhs = thrust::get<0>(equations);
	Agent rhs = thrust::get<1>(equations);

	// default: no interaction was performed
	thrust::get<8>(equations) = false;

		
	// if both sides are agents
	// rewrite the equations
	//TODO replace is_upper by direct char comparison
	if (is_upper(lhs.name) && is_upper(rhs.name)) {

		//sort pair lexicographically (active pairs are symmetric)
		if (rhs.name < lhs.name ) {
			lhs = rhs;
			rhs = thrust::get<0>(equations);
		}

		// A(s) >< L(r,i) => M(r,s)~i;
		if (lhs.name == 'A' && rhs.name == 'L') {
			// wires
			const int s = lhs.ports[0];
			const int r = rhs.ports[0];
			const int i = rhs.ports[1];
			// M(r,s)~i
			get<0>(equations) = Agent(i, 'i', 0);
			get<1>(equations) = Agent(rhs.id, 'M', 2, r, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// B(s) >< L(r,i) => N(r,s)~i;
		else if (lhs.name == 'B' && rhs.name == 'L') {
			// wires
			const int s = lhs.ports[0];
			const int r = rhs.ports[0];
			const int i = rhs.ports[1];
			// M(r,s)~i
			get<0>(equations) = Agent(i, 'i', 0);
			get<1>(equations) = Agent(lhs.id, 'N', 2, r, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// M(r,s) >< O => r~A(s)
		else if (lhs.name == 'M' && rhs.name == 'O') {
			// wires
			const int r = lhs.ports[0];
			const int s = lhs.ports[1];
			// r~A(s)
			get<0>(equations) = Agent(r);
			get<1>(equations) = Agent(lhs.id, 'A', 1, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// N(r,s) >< O => r~B(s)
		else if (lhs.name == 'N' && rhs.name == 'O') {
			// wires
			const int r = lhs.ports[0];
			const int s = lhs.ports[1];
			// r~A(s)
			get<0>(equations) = Agent(r);
			get<1>(equations) = Agent(lhs.id, 'B', 1, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// M(r,s) >< S(i) => L(r,d)~A(y), L(y,e)~B(s), D(d,e)~i
		else if (lhs.name == 'M' && rhs.name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], algae_max_idx);
			// wires
			const int r = lhs.ports[0];
			const int s = lhs.ports[1];
			const int i = rhs.ports[0];
			const int d = new_index++;
			const int y = new_index++;
			const int e = new_index++;
			//  L(r,d)~A(y)
			get<0>(equations) = Agent(new_index++, 'L', 2, r, d);
			get<1>(equations) = Agent(new_index++, 'A', 1, y);
			//  L(y,e)~B(s)
			get<2>(equations) = Agent(new_index++, 'L', 2, y, e);
			get<3>(equations) = Agent(new_index++, 'B', 1, s);
			// D(d,e)~i
			get<4>(equations) = Agent(i);
			get<5>(equations) = Agent(new_index++, 'D', 2, d, e);

			get<8>(equations) = true; // we performed an interaction
		}
		// N(r,s) >< S(i) => L(r,x)~A(s)
		else if (lhs.name == 'N' && rhs.name == 'S') {
			// wires
			const int r = lhs.ports[0];
			const int s = lhs.ports[1];
			const int i = rhs.ports[0];
			// L(r,x)~A(s)
			get<0>(equations) = Agent(lhs.id, 'L', 2, r, i);
			get<1>(equations) = Agent(rhs.id, 'A', 1, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// O >< d(x,y) => O~x, O~y;
		else if (lhs.name == 'D' && rhs.name == 'O' ) {
			int new_index = atomicAdd(&index_ptr[0], 1); 
			// eps~x
			thrust::get<1>(equations) = rhs;
			thrust::get<0>(equations) = Agent(lhs.ports[0], 'v', 0);//*(rhs.ports[0].get());
			// eps~y goes to equations array A
			thrust::get<2>(equations) = Agent(lhs.ports[1], 'v', 0);//*(rhs.ports[1].get());
			thrust::get<3>(equations) = Agent(new_index, 'O', 0);

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
		// S(z) >< d(x,y) => S(a)~x, S(b)~y, d(a,b)~z;
		else if (lhs.name == 'D' && rhs.name == 'S' ) {
			int new_index = atomicAdd(&index_ptr[0], 4); 
			//wires
			const int x = lhs.ports[0];
			const int y = lhs.ports[1];
			const int z = rhs.ports[0];
			const int a = new_index++;
			const int b = new_index++;
			rhs.ports[0] = a;
			Agent s(new_index++, 'S', 1);
			s.ports[0] = b;
			lhs.ports[0] = a;
			lhs.ports[1] = b;
			// S(a)~x
			thrust::get<1>(equations) = rhs;
			thrust::get<0>(equations) = Agent(x, 'x', 0);//*(rhs.ports[0].get());
			// S(b)~y
			thrust::get<3>(equations) = s;
			thrust::get<2>(equations) = Agent(y, 'y', 0);//*(rhs.ports[1].get());
			// d(a,b)~ z
			get<4>(equations) = Agent(z, 'v', 0);
			get<5>(equations) = lhs;

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
	}
}
};

#endif