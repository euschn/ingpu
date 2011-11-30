// algae functor
// computes nth iteration of original L-system to model algae groth
// en.wikipedia.org/wiki/L-system

//IMPORTANT
// RHS equations must be oriented


#pragma once

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "agent_functor.h"

using namespace soa;
using namespace thrust;

namespace soa {

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
	agent lhs = thrust::get<0>(equations);
	agent rhs = thrust::get<1>(equations);

	// default: no interaction was performed
	thrust::get<8>(equations) = false;

		
	// if both sides are agents
	// rewrite the equations
	if (is_upper( get<NAME>(lhs) ) && is_upper( get<NAME>(lhs) )) {

		//sort pair lexicographically (active pairs are symmetric)
		if (get<NAME>(rhs) < get<NAME>(lhs) ) {
			lhs = rhs;
			rhs = thrust::get<0>(equations);
		}
		
		name_type lhs_name = get<NAME>(lhs);
		name_type rhs_name = get<NAME>(rhs);

		// A(s) >< L(r,i) => M(r,s)~i;
		if (lhs_name == 'A' && rhs_name == 'L') {
			int new_index = atomicAdd(&index_ptr[0], 1);
			// wires
			const int s = get<P1>(lhs);
			const int r = get<P1>(rhs);
			const int i = get<P2>(rhs);
			// M(r,s)~i
			get<0>(equations) = make_agent(i, 'i', 0);
			get<1>(equations) = make_agent(new_index, 'M', 2, r, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// B(s) >< L(r,i) => N(r,s)~i;
		else if (lhs_name == 'B' && rhs_name == 'L') {
			int new_index = atomicAdd(&index_ptr[0], 1);
			// wires
			const int s = get<P1>(lhs);
			const int r = get<P1>(rhs);
			const int i = get<P2>(rhs);
			// M(r,s)~i
			get<0>(equations) = make_agent(i, 'i', 0);
			get<1>(equations) = make_agent(new_index, 'N', 2, r, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// M(r,s) >< O => r~A(s)
		else if (lhs_name == 'M' && rhs_name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], 1);
			// wires
			const int r = get<P1>(lhs);
			const int s = get<P2>(lhs);
			// r~A(s)
			get<0>(equations) = make_agent(r);
			get<1>(equations) = make_agent(new_index, 'A', 1, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// N(r,s) >< O => r~B(s)
		else if (lhs_name == 'N' && rhs_name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], 1);
			// wires
			const int r = get<P1>(lhs);
			const int s = get<P2>(lhs);
			// r~A(s)
			get<0>(equations) = make_agent(r);
			get<1>(equations) = make_agent(new_index, 'B', 1, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// M(r,s) >< S(i) => L(r,d)~A(y), L(y,e)~B(s), D(d,e)~i
		else if (lhs_name == 'M' && rhs_name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], algae_max_idx);
			// wires
			const int r = get<P1>(lhs);
			const int s = get<P2>(lhs);
			const int i = get<P1>(rhs);
			const int d = new_index++;
			const int y = new_index++;
			const int e = new_index++;
			//  L(r,d)~A(y)
			get<0>(equations) = make_agent(new_index++, 'L', 2, r, d);
			get<1>(equations) = make_agent(new_index++, 'A', 1, y);
			//  L(y,e)~B(s)
			get<2>(equations) = make_agent(new_index++, 'L', 2, y, e);
			get<3>(equations) = make_agent(new_index++, 'B', 1, s);
			// D(d,e)~i
			get<4>(equations) = make_agent(i);
			get<5>(equations) = make_agent(new_index++, 'D', 2, d, e);

			get<8>(equations) = true; // we performed an interaction
		}
		// N(r,s) >< S(i) => L(r,x)~A(s)
		else if (lhs_name == 'N' && rhs_name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], 2);
			// wires
			const int r = get<P1>(lhs);
			const int s = get<P2>(lhs);
			const int i = get<P1>(rhs);
			// L(r,x)~A(s)
			get<0>(equations) = make_agent(new_index++, 'L', 2, r, i);
			get<1>(equations) = make_agent(new_index++, 'A', 1, s);

			get<8>(equations) = true; // we performed an interaction
		}
		// O >< d(x,y) => O~x, O~y;
		else if (lhs_name == 'D' && rhs_name == 'O' ) {
			int new_index = atomicAdd(&index_ptr[0], algae_max_idx); 
			// eps~x
			thrust::get<1>(equations) = rhs;
			thrust::get<0>(equations) = make_agent(get<P1>(lhs), 'z', 0);//*(get<P1>(rhs).get());
			// eps~y goes to equations array A
			thrust::get<3>(equations) = make_agent(new_index, 'O', 0);
			thrust::get<2>(equations) = make_agent(get<P2>(lhs), 'z', 0);//*(get<P2>(rhs).get());

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
		// S(z) >< d(x,y) => S(a)~x, S(b)~y, d(a,b)~z;
		else if (lhs_name == 'D' && rhs_name == 'S' ) {
			int new_index = atomicAdd(&index_ptr[0], algae_max_idx); 
			//wires
			const int x = get<P1>(lhs);
			const int y = get<P2>(lhs);
			const int z = get<P1>(rhs);
			const int a = new_index++;
			const int b = new_index++;
			//get<P1>(rhs) = a;
			//agent s = ;
			//get<P1>(s) = b;
			//get<P1>(lhs) = a;
			//get<P2>(lhs) = b;
			// S(a)~x
			thrust::get<1>(equations) = make_agent(new_index++, 'S', 1, a);
			thrust::get<0>(equations) = make_agent(x, 'x', 0);//*(get<P1>(rhs).get());
			// S(b)~y
			thrust::get<3>(equations) = make_agent(new_index++, 'S', 1, b);
			thrust::get<2>(equations) = make_agent(y, 'y', 0);//*(get<P2>(rhs).get());
			// d(a,b)~ z
			get<4>(equations) = make_agent(z, 'z', 0);
			get<5>(equations) = make_agent(new_index++, 'D', 2, a, b);

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
	}
}
};

}