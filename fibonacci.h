// fibonacci functor


//#pragma once

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "agent.cuh"

using namespace ingpu;
using namespace thrust;

//max new indices per interaction
const int max_idx = 7;

struct fibonacci_functor : public int_functor {
	
int * index_ptr;
__host__ __device__
fibonacci_functor(int * ptr) : index_ptr(ptr) {}

template <typename Tuple>
__device__
void operator()(Tuple equations)
{
	
	//get lhs and rhs of the equations
	Agent lhs = get<0>(equations);
	Agent rhs = get<1>(equations);

	// default: no interaction was performed
	get<8>(equations) = false;

		
	// if both sides are agents
	// rewrite the equations
	if (is_upper(lhs.name) && is_upper(rhs.name)) {

		//sort pair lexicographically (active pairs are symmetric)
		if (rhs.name < lhs.name ) {
			lhs = rhs;
			rhs = thrust::get<0>(equations);
		}
		
		// A(r, y) >< 0 => r~y
		if (lhs.name == 'A' && rhs.name == 'O') {
	
			//r~y
			thrust::get<0>(equations) = Agent(lhs.ports[0], 'r', 0);
			thrust::get<1>(equations) = Agent(lhs.ports[1], 'y', 0);

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// A(r, y) >< S(x) => r~S(w), A(w, y)~x
		else if (lhs.name == 'A' && rhs.name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], max_idx); 
			//wires
			int x = rhs.ports[0];
			int r = lhs.ports[0];
			int w = new_index; // new variable w
	
			// r~S(w)
			rhs.ports[0] = w;
			thrust::get<0>(equations) = Agent(r, 'r', 0);
			thrust::get<1>(equations) = rhs;
			// A(w, y)~x
			lhs.ports[0] = w;
			thrust::get<2>(equations) = Agent(x, 'x', 0);
			thrust::get<3>(equations) = lhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib(r) >< O => r~O;
		else if (lhs.name == 'F' && rhs.name == 'O') {
			// wires
			const int r = lhs.ports[0];
			//r~O
			get<0>(equations) = Agent(r, 'r', 0);
			get<1>(equations) = rhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib(r) >< S(x) => fib2(r)~x;
		else if (lhs.name == 'F' && rhs.name == 'S') {
			// wires
			const int r = lhs.ports[0];
			const int x = rhs.ports[0];
			Agent g(lhs.id, 'G', 1);
			g.ports[0] = r;
			// G(r)~x
			get<0>(equations) = Agent(x, 'x', 0);
			get<1>(equations) = g;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib2(r) >< O => r~S(a), a~O;
		else if (lhs.name == 'G' && rhs.name == 'O') {
			int new_index = atomicAdd(&index_ptr[0], max_idx); 
			// wires
			const int r = lhs.ports[0];
			const int a = new_index;
			Agent s(lhs.id, 'S', 1);
			s.ports[0] = a;
			//r~S(a)
			get<0>(equations) = Agent(r, 'r',0);
			get<1>(equations) = s;
			//a~O
			get<2>(equations) = Agent(a, 'a',0);
			get<3>(equations) = rhs;


			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib-(r) >< O => r~O;
		else if (lhs.name == 'H' && rhs.name == 'O') {
			// wires
			const int r = lhs.ports[0];
			//r~O
			get<0>(equations) = Agent(r, 'r', 0);
			get<1>(equations) = rhs;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib-(r) >< S(x) => fib(r)~x;
		else if (lhs.name == 'H' && rhs.name == 'S') {
			// wires
			const int r = lhs.ports[0];
			const int x = rhs.ports[0];
			Agent g(lhs.id, 'F', 1);
			g.ports[0] = r;
			// G(r)~x
			get<0>(equations) = Agent(x, 'x', 0);
			get<1>(equations) = g;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// fib2(r) >< S(x) => add(r,a)~b, fib-(a)~d, fib(b)~e, D(d,e)~x
		else if (lhs.name == 'G' && rhs.name == 'S') {
			int new_index = atomicAdd(&index_ptr[0], max_idx); 
			// wires
			const int r = lhs.ports[0];
			const int x = rhs.ports[0];
			const int a = new_index;
			const int b = new_index + 1;
			const int d = new_index + 2;
			const int e = new_index + 3;
			Agent add(lhs.id, 'A', 2);
			add.ports[0] = r;
			add.ports[1] = a;
			Agent fibmin(rhs.id, 'G', 1);
			fibmin.ports[0] = a;
			Agent fib(new_index + 4, 'F', 1);
			fib.ports[0] = b;
			Agent delta(new_index + 5, 'D', 2);
			delta.ports[0] = d;
			delta.ports[1] = e;

			// add(r,a)~b
			get<0>(equations) = Agent(b);
			get<1>(equations) = add;
			//fib-(a)~d
			get<2>(equations) = Agent(d);
			get<3>(equations) = fibmin;
			// fib(b), e
			get<4>(equations) = Agent(e);
			get<5>(equations) = fib;
			// D(d,e)~x
			get<6>(equations) = Agent(x);
			get<7>(equations) = delta;

			thrust::get<8>(equations) = true; // we performed an interaction
		}
		// O >< d(x,y) => O~x, O~y;
		else if (lhs.name == 'D' && rhs.name == 'O' ) {
			int new_index = atomicAdd(&index_ptr[0], max_idx); 
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
			int new_index = atomicAdd(&index_ptr[0], max_idx); 
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