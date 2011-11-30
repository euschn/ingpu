// ackermann functor soa version


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
const int ackermann_max_idx = 6;

struct ackermann_functor : public int_functor {

int * index_ptr;
__host__ __device__
ackermann_functor(int * ptr) : index_ptr(ptr) {}

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

		// P(r) = S(x) => r~x;
		if (lhs_name == 'P' && rhs_name == 'S') {
			// wires
			const int r = get<P1>(lhs);
			const int x = get<P1>(rhs);
			// r~x
			get<0>(equations) = make_agent(r, 'r', 0);
			get<1>(equations) = make_agent(x, 'x', 0);

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// A(r,n) >< O => r~S(n);
		else if (lhs_name == 'A' && rhs_name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			agent s = make_agent(new_index, 'S', 1);
			//wires
			const int r = get<P1>(lhs);
			const int n = get<P2>(lhs);
			get<P1>(s)  = n;
			// r~S(n)
			get<0>(equations) = make_agent(r, 'r', 0);
			get<1>(equations) = s;

			get<8>(equations) = true; // we performed an interaction
		}
		// A(r,n) >< S(m) => B(r,x) = n, x = S(m)
		else if (lhs_name == 'A' && rhs_name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			agent b(get<ID>(lhs), 'B', 2);
			agent x = make_agent(new_index, 'x', 0);
			//wires
			const int r = get<P1>(lhs);
			const int n = get<P2>(lhs);
			//b.ports[0] = r;
			//b.ports[1] = x);

			// B(r,x) = n
			get<1>(equations) = make_agent(get<ID>(lhs), 'B', 2, r, get<ID>(x), 0 , 0);
			get<0>(equations)  = agent(n, 'n', 0);
			// x = s(m)
			get<2>(equations) = x;
			get<3>(equations) = rhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// B(r,m) >< O => A(r, x) = y, P(y) = m, x = S(z), z = O
		else if (lhs_name == 'B' && rhs_name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			agent x = make_agent(new_index, 'x', 0);
			agent y = make_agent(new_index + 1, 'y', 0);
			agent z = make_agent(new_index + 2, 'z', 0);
			agent p = make_agent(new_index + 3, 'P', 1);
			agent s = make_agent(new_index + 4, 'S', 1);
			agent a = make_agent(get<ID>(lhs), 'A', 2);
			//wires
			const int r = get<P1>(lhs);
			const int m = get<P2>(lhs);
			get<P1>(a) = r;
			get<P2>(a) = get<ID>(x);
			get<P1>(p) = get<ID>(y);
			get<P1>(s) = get<ID>(z);

			// y = A(r,x);
			get<0>(equations) = y;
			get<1>(equations) = a;
			// m = P(y)
			get<2>(equations) = make_agent(m, 'm', 0);
			get<3>(equations) = p;
			// x = S(z)
			get<4>(equations) = x;
			get<5>(equations) = s;
			// z = 0
			get<6>(equations) = z;
			get<7>(equations) = rhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// B(r,m) >< S(n) => A(r,s)~x, P(x)~d, A(s,n)~e, D(d,e)~m;
		else if (lhs_name == 'B' && rhs_name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx);
			agent p = make_agent(get<ID>(rhs), 'P', 1);
			agent a1 = make_agent(get<ID>(lhs), 'A', 2);
			agent x = make_agent(new_index, 'x', 0);
			agent s = make_agent(new_index + 1, 's', 0);
			agent d = make_agent(new_index + 2, 'd', 0);
			agent e = make_agent(new_index + 3, 'e', 0);
			agent a2 = make_agent(new_index + 4, 'A', 2);
			agent delta = make_agent(new_index + 5, 'D', 2);
			//wires
			const int r = get<P1>(lhs);
			const int m = get<P2>(lhs);
			const int n = get<P1>(rhs);
			get<P1>(a1) = r;
			get<P2>(a1) = get<ID>(s);
			get<P1>(p) = get<ID>(x);
			get<P1>(a2) = get<ID>(s);
			get<P2>(a2) = n;
			get<P1>(delta) = get<ID>(d);
			get<P2>(delta) = get<ID>(e);
			// A(r,s) ~ x
			get<1>(equations) = a1;
			get<0>(equations) = x;
			// P(x) ~ d
			get<3>(equations) = p;
			get<2>(equations) = d;
			// A(s,n) ~ e
			get<5>(equations) = a2;
			get<4>(equations) = e;
			// D(d,e) ~ m
			get<6>(equations) = make_agent(m, 'm', 0);
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
			agent s = make_agent(new_index, 'S', 1);
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
