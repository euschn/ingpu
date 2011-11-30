// cantor_dust functor
// computes cantor fractal set on a real line
// en.wikipedia.org/wiki/L-system
// A -> ABA
// B -> BBB

//IMPORTANT
// RHS equations must be oriented

//#pragma once
#ifndef CANTOR_DUST_H_
#define CANTOR_DUST_H_

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
const int cantor_dust_max_idx = 9;

struct cantor_dust_functor : public int_functor {

int * index_ptr;
__host__ __device__
cantor_dust_functor(int * ptr) : index_ptr(ptr) {}


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
		// M(r,s) >< S(i) => L(r,d)~A(y), L(y,e)~B(z), L(z,f)~A(s), D(d,e,f)~i
		else if (lhs.name == 'M' && rhs.name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], 11);
			// wires
			const int r = lhs.ports[0];
			const int s = lhs.ports[1];
			const int i = rhs.ports[0];
			const int d = new_index++;
			const int e = new_index++;
			const int f = new_index++;
			const int y = new_index++;
			const int z = new_index++;
			//  L(r,d)~A(y)
			get<0>(equations) = Agent(new_index++, 'L', 2, r, d);
			get<1>(equations) = Agent(new_index++, 'A', 1, y);
			//  L(y,e)~B(z)
			get<2>(equations) = Agent(new_index++, 'L', 2, y, e);
			get<3>(equations) = Agent(new_index++, 'B', 1, z);
			// L(z,f)~A(s)
			get<4>(equations) = Agent(new_index++, 'L', 2, z, f);
			get<5>(equations) = Agent(new_index++, 'A', 1, s);
			// D(d,e)~i
			get<6>(equations) = Agent(i);
			get<7>(equations) = Agent(new_index++, 'D', 3, d, e, f);

			get<8>(equations) = true; // we performed an interaction
		}
		// N(r,s) >< S(i) => L(r,d)~B(y), L(y,e)~B(z), L(z,f)~B(s), D(d,e,f)~i
		else if (lhs.name == 'N' && rhs.name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], 11);
			// wires
			const int r = lhs.ports[0];
			const int s = lhs.ports[1];
			const int i = rhs.ports[0];
			const int d = new_index++;
			const int e = new_index++;
			const int f = new_index++;
			const int y = new_index++;
			const int z = new_index++;
			//  L(r,d)~A(y)
			get<0>(equations) = Agent(new_index++, 'L', 2, r, d);
			get<1>(equations) = Agent(new_index++, 'B', 1, y);
			//  L(y,e)~B(z)
			get<2>(equations) = Agent(new_index++, 'L', 2, y, e);
			get<3>(equations) = Agent(new_index++, 'B', 1, z);
			// L(z,f)~A(s)
			get<4>(equations) = Agent(new_index++, 'L', 2, z, f);
			get<5>(equations) = Agent(new_index++, 'B', 1, s);
			// D(d,e)~i
			get<6>(equations) = Agent(i);
			get<7>(equations) = Agent(new_index++, 'D', 3, d, e, f);

			get<8>(equations) = true; // we performed an interaction
		}
		// O >< D(x,y,z) => O~x, O~y, O~z;
		else if (lhs.name == 'D' && rhs.name == 'O' ) {
			int new_index = atomicAdd(&index_ptr[0], 2); 
			// O~x
			thrust::get<1>(equations) = rhs;
			thrust::get<0>(equations) = Agent(lhs.ports[0], 'v', 0);//*(rhs.ports[0].get());
			// O~y
			thrust::get<3>(equations) = Agent(new_index++, 'O', 0);
			thrust::get<2>(equations) = Agent(lhs.ports[1], 'v', 0);//*(rhs.ports[1].get());
			// O~z
			thrust::get<5>(equations) = Agent(new_index, 'O', 0);
			thrust::get<4>(equations) = Agent(lhs.ports[2], 'v', 0);//*(rhs.ports[0].get());

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
		// S(n) >< D(x,y,z) => S(a)~x, S(b)~y, S(c), D(a,b,c)~n;
		else if (lhs.name == 'D' && rhs.name == 'S' ) {
			int new_index = atomicAdd(&index_ptr[0], 5); 
			//wires
			const int x = lhs.ports[0];
			const int y = lhs.ports[1];
			const int z = lhs.ports[2];
			const int n = rhs.ports[0];
			const int a = new_index++;
			const int b = new_index++;
			const int c = new_index++;
			rhs.ports[0] = a;
			Agent s(new_index++, 'S', 1);
			s.ports[0] = b;
			lhs.ports[0] = a;
			lhs.ports[1] = b;
			lhs.ports[2] = c;
			// S(a)~x
			thrust::get<1>(equations) = rhs;
			thrust::get<0>(equations) = Agent(x, 'x', 0);//*(rhs.ports[0].get());
			// S(b)~y
			thrust::get<3>(equations) = s;
			thrust::get<2>(equations) = Agent(y, 'y', 0);//*(rhs.ports[1].get());
			// S(c)~z
			thrust::get<5>(equations) = Agent(new_index++, 'S', 1, c);
			thrust::get<4>(equations) = Agent(z, 'z', 0);//*(rhs.ports[1].get());
			// D(a,b,c)~ n
			get<6>(equations) = Agent(n, 'v', 0);
			get<7>(equations) = lhs;

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
	}
}
};

#endif