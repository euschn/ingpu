// fibonacci functor


//#pragma once

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "agent_functor.h"

using namespace soa;
using namespace thrust;

//max new indices per interaction
const int max_idx = 7;

namespace soa {

struct fibonacci_functor : public int_functor {

int * index_ptr;
__host__ __device__
fibonacci_functor(int * ptr) : index_ptr(ptr) {}

template <typename Tuple>
__device__
void operator()(Tuple equations)
{
	
	//get lhs and rhs of the equations
	agent lhs = get<0>(equations);
	agent rhs = get<1>(equations);

	// default: no interaction was performed
	get<8>(equations) = false;

		
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
		
		// A(r, y) >< 0 => r~y
		if (lhs_name == 'A' && rhs_name == 'O') {
	
			//r~y
			thrust::get<0>(equations) = make_agent(get<P1>(lhs), 'r', 0);
			thrust::get<1>(equations) = make_agent(get<P2>(lhs), 'y', 0);

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// A(r, y) >< S(x) => r~S(w), A(w, y)~x
		else if (lhs_name == 'A' && rhs_name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			//wires
			int x = get<P1>(rhs);
			int r = get<P1>(lhs);
			int w = new_index; // new variable w
	
			// r~S(w)
			get<P1>(rhs) = w;
			thrust::get<0>(equations) = make_agent(r, 'r', 0);
			thrust::get<1>(equations) = rhs;
			// A(w, y)~x
			get<P1>(lhs) = w;
			thrust::get<2>(equations) = make_agent(x, 'x', 0);
			thrust::get<3>(equations) = lhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib(r) >< O => r~O;
		else if (lhs_name == 'F' && rhs_name == 'O') {
			// wires
			const int r = get<P1>(lhs);
			//r~O
			get<0>(equations) = make_agent(r, 'r', 0);
			get<1>(equations) = rhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib(r) >< S(x) => fib2(r)~x;
		else if (lhs_name == 'F' && rhs_name == 'S') {
			// wires
			const int r = get<P1>(lhs);
			const int x = get<P1>(rhs);
			agent g = make_agent(get<ID>(lhs), 'G', 1);
			get<P1>(g) = r;
			// G(r)~x
			get<0>(equations) = make_agent(x, 'x', 0);
			get<1>(equations) = g;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib2(r) >< O => r~S(a), a~O;
		else if (lhs_name == 'G' && rhs_name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			// wires
			const int r = get<P1>(lhs);
			const int a = new_index;
			agent s(get<ID>(lhs), 'S', 1);
			get<P1>(s) = a;
			//r~S(a)
			get<0>(equations) = make_agent(r, 'r',0);
			get<1>(equations) = s;
			//a~O
			get<2>(equations) = make_agent(a, 'a',0);
			get<3>(equations) = rhs;


			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib-(r) >< O => r~O;
		else if (lhs_name == 'H' && rhs_name == 'O') {
			// wires
			const int r = get<P1>(lhs);
			//r~O
			get<0>(equations) = make_agent(r, 'r', 0);
			get<1>(equations) = rhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib-(r) >< S(x) => fib(r)~x;
		else if (lhs_name == 'H' && rhs_name == 'S') {
			// wires
			const int r = get<P1>(lhs);
			const int x = get<P1>(rhs);
			agent g(get<ID>(lhs), 'F', 1);
			get<P1>(g) = r;
			// G(r)~x
			get<0>(equations) = make_agent(x, 'x', 0);
			get<1>(equations) = g;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib2(r) >< S(x) => add(r,a)~b, fib-(a)~d, fib(b)~e, D(d,e)~x
		else if (lhs_name == 'G' && rhs_name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			// wires
			const int r = get<P1>(lhs);
			const int x = get<P1>(rhs);
			const int a = new_index;
			const int b = new_index + 1;
			const int d = new_index + 2;
			const int e = new_index + 3;
			agent add(get<ID>(lhs), 'A', 2);
			get<P1>(add) = r;
			get<P2>(add) = a;
			agent fibmin = make_agent(get<ID>(rhs), 'G', 1);
			get<P1>(fibmin) = a;
			agent fib = make_agent(new_index + 4, 'F', 1);
			get<P1>(fib) = b;
			agent delta(new_index + 5, 'D', 2);
			get<P1>(delta) = d;
			get<P2>(delta) = e;

			// add(r,a)~b
			get<0>(equations) = make_agent(b);
			get<1>(equations) = add;
			//fib-(a)~d
			get<2>(equations) = make_agent(d);
			get<3>(equations) = fibmin;
			// fib(b), e
			get<4>(equations) = make_agent(e);
			get<5>(equations) = fib;
			// D(d,e)~x
			get<6>(equations) = make_agent(x);
			get<7>(equations) = delta;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// O >< d(x,y) => O~x, O~y;
		else if (lhs_name == 'D' && rhs_name == 'O' ) {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
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
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			//wires
			const int x = get<P1>(lhs);
			const int y = get<P2>(lhs);
			const int z = get<P1>(rhs);
			const int a = new_index + 1;
			const int b = new_index + 2;
			get<P1>(rhs) = a;
			agent s(new_index, 'S', 1);
			get<P1>(s) = b;
			get<P1>(lhs) = a;
			get<P2>(lhs) = b;
			// S(a)~x
			thrust::get<1>(equations) = rhs;
			thrust::get<0>(equations) = make_agent(x, 'x', 0);//*(get<P1>(rhs).get());
			// S(b)~y
			thrust::get<3>(equations) = s;
			thrust::get<2>(equations) = make_agent(y, 'y', 0);//*(get<P2>(rhs).get());
			// d(a,b)~ z
			get<4>(equations) = make_agent(z, 'z', 0);
			get<5>(equations) = lhs;

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
	}
}
};

}