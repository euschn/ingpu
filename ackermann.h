// ackermann functor


//#pragma once
#ifndef ACKERMANN_H_
#define ACKERMANN_H_

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
	Agent lhs = thrust::get<0>(equations);
	Agent rhs = thrust::get<1>(equations);

	// default: no interaction was performed
	thrust::get<8>(equations) = false;

		
	// if both sides are agents
	// rewrite the equations
	if (is_upper(lhs.name) && is_upper(rhs.name)) {

		//sort pair lexicographically (active pairs are symmetric)
		if (rhs.name < lhs.name ) {
			lhs = rhs;
			rhs = thrust::get<0>(equations);
		}

		// P(r) = S(x) => r~x;
		if (lhs.name == 'P' && rhs.name == 'S') {
			// wires
			const int r = lhs.ports[0];
			const int x = rhs.ports[0];
			// r~x
			thrust::get<0>(equations) = Agent(r, 'r', 0);
			thrust::get<1>(equations) = Agent(x, 'x', 0);

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// A(r,n) >< O => r~S(n);
		else if (lhs.name == 'A' && rhs.name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx); 
			Agent s(new_index, 'S', 1);
			//wires
			const int r = lhs.ports[0];
			const int n = lhs.ports[1];
			s.ports[0]  = n;
			// r~S(n)
			thrust::get<0>(equations) = Agent(r, 'r', 0);
			thrust::get<1>(equations) = s;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// A(r,n) >< S(m) => B(r,x) = n, x = S(m)
		else if (lhs.name == 'A' && rhs.name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx); 
			Agent b(lhs.id, 'B', 2);
			Agent x(new_index, 'x', 0);
			//wires
			const int r = lhs.ports[0];
			const int n = lhs.ports[1];
			b.ports[0] = r;
			b.ports[1] = x.id;

			// B(r,x) = n
			get<1>(equations) = b;
			get<0>(equations)  = Agent(n, 'n', 0);
			// x = s(m)
			get<2>(equations) = x;
			get<3>(equations) = rhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// B(r,m) >< O => A(r, x) = y, P(y) = m, x = S(z), z = O
		else if (lhs.name == 'B' && rhs.name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx); 
			Agent x(new_index, 'x', 0);
			Agent y(new_index + 1, 'y', 0);
			Agent z(new_index + 2, 'z', 0);
			Agent p(new_index + 3, 'P', 1);
			Agent s(new_index + 4, 'S', 1);
			Agent a(lhs.id, 'A', 2);
			//wires
			const int r = lhs.ports[0];
			const int m = lhs.ports[1];
			a.ports[0] = r;
			a.ports[1] = x.id;
			p.ports[0] = y.id;
			s.ports[0] = z.id;

			// y = A(r,x);
			get<0>(equations) = y;
			get<1>(equations) = a;
			// m = P(y)
			get<2>(equations) = Agent(m, 'm', 0);
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
		else if (lhs.name == 'B' && rhs.name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx); 
			Agent p(rhs.id, 'P', 1);
			Agent a1(lhs.id, 'A', 2);
			Agent x(new_index, 'x', 0);
			Agent s(new_index + 1, 's', 0);
			Agent d(new_index + 2, 'd', 0);
			Agent e(new_index + 3, 'e', 0);
			Agent a2(new_index + 4, 'A', 2);
			Agent delta(new_index + 5, 'D', 2);
			//wires
			const int r = lhs.ports[0];
			const int m = lhs.ports[1];
			const int n = rhs.ports[0];
			a1.ports[0] = r;
			a1.ports[1] = s.id;
			p.ports[0] = x.id;
			a2.ports[0] = s.id;
			a2.ports[1] = n;
			delta.ports[0] = d.id;
			delta.ports[1] = e.id;
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
			get<6>(equations) = Agent(m, 'm', 0);
			get<7>(equations) = delta;
			

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// O >< d(x,y) => O~x, O~y;
		else if (lhs.name == 'D' && rhs.name == 'O' ) {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx); 
			// eps~x
			thrust::get<1>(equations) = rhs;
			thrust::get<0>(equations) = Agent(lhs.ports[0], 'z', 0);//*(rhs.ports[0].get());
			// eps~y goes to equations array A
			thrust::get<3>(equations) = Agent(new_index, 'O', 0);
			thrust::get<2>(equations) = Agent(lhs.ports[1], 'z', 0);//*(rhs.ports[1].get());

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
		// S(z) >< d(x,y) => S(a)~x, S(b)~y, d(a,b)~z;
		else if (lhs.name == 'D' && rhs.name == 'S' ) {
			int new_index = atomicAdd(&index_ptr[0], ackermann_max_idx); 
			//wires
			const int x = lhs.ports[0];
			const int y = lhs.ports[1];
			const int z = rhs.ports[0];
			const int a = new_index + 1;
			const int b = new_index + 2;
			rhs.ports[0] = a;
			Agent s(new_index, 'S', 1);
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
			get<4>(equations) = Agent(z, 'z', 0);
			get<5>(equations) = lhs;

			thrust::get<8>(equations) = true;	// we performed an interaction
		}
	}
}
};

void build_ack_input(thrust::host_vector<Agent>& in_lhs, thrust::host_vector<Agent>& in_rhs)
{
	in_lhs[0] = Agent(1, 'S', 1);
	in_rhs[0] = Agent(2, 'x', 0);

	in_lhs[1] = Agent(3, 'A', 2);
	in_lhs[1].ports[0] = 0;
	in_rhs[1] = Agent(4, 'S', 1);
	in_lhs[1].ports[1] = in_rhs[0].id;

	in_lhs[2] = Agent(5, 'O', 0);
	in_rhs[2] = Agent(6, 'y', 0);
	in_rhs[1].ports[0] = in_rhs[2].id;

	in_lhs[3] = Agent(7, 'O', 0);
	in_rhs[3] = Agent(8, 'y', 0);
	in_lhs[0].ports[0] = in_rhs[3].id;
}

#endif