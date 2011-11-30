#include "kernel_soa.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iterator>
#include <cstdlib>
#include <sstream>
#include <conio.h>
#include <assert.h>


#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust\logical.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <cutil_inline.h>
#include <cutil.h>
#include <stopwatch.h>
#include <stopwatch_win.h>
//#include "agent.h"
#include "ackermann.h"
#include "fibonacci.h"
#include "algae.h"
#include "agent_functor.h"
#include "communication.h"


using namespace thrust;
using namespace soa;

namespace soa {
	//print a vector
	template <typename Iterator>
	void print_range(const std::string& name, Iterator first, Iterator last)
	{
		typedef typename std::iterator_traits<Iterator>::value_type T;

		std::cout << name << ": ";
		thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
		std::cout << std::endl;
	}
}

//void resize_input(const int size)
//{	
//	lhs_ids.resize(size); lhs_arities.resize(size); lhs_names.resize(size);
//	lhs_p1s.resize(size); lhs_p2s.resize(size); lhs_p3s.resize(size); lhs_p4s.resize(size);
//	rhs_ids.resize(size); rhs_arities.resize(size); rhs_names.resize(size);
//	rhs_p1s.resize(size); rhs_p2s.resize(size); rhs_p3s.resize(size); rhs_p4s.resize(size);
//}

//clear the additional result arrays
//TODO only clear indices to speedup
struct clear_functor {

template <typename Tuple>
__host__ __device__
void operator()(Tuple& equations)
{
	thrust::get<0>(equations) = thrust::make_tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type>(0, 'Z', 0, 0, 0, 0, 0);
	thrust::get<1>(equations) = thrust::make_tuple<id_type, name_type, arity_type, port_type, port_type, port_type, port_type>(0, 'Z', 0, 0, 0, 0, 0);
}

};


/*-------------------------
Interaction Nets Evaluation
-------------------------*/
result_info soa::interaction_loop(
	const int ruleset, 
	host_vector<int>& host_lhs_ids,
	host_vector<char>& host_lhs_names,
	host_vector<int>& host_lhs_arities,
	host_vector<int>& host_lhs_p1s,
	host_vector<int>& host_lhs_p2s,
	host_vector<int>& host_lhs_p3s,
	host_vector<int>& host_lhs_p4s,
	host_vector<int>& host_rhs_ids,
	host_vector<char>& host_rhs_names,
	host_vector<int>& host_rhs_arities,
	host_vector<int>& host_rhs_p1s,
	host_vector<int>& host_rhs_p2s,
	host_vector<int>& host_rhs_p3s,
	host_vector<int>& host_rhs_p4s,
	const bool verbose, const bool print_final, const bool pause)
{
	// --------------
	// initialization
	// --------------
	//input size
	int input_size = host_lhs_ids.size();
	//the full size of the vector, including virtual auxiliary vectors
	int vector_size = input_size * 4;
	//auxiliary vector size
	int aux_input_size = input_size;

	//make a device pointer for tracking agent indices
	//------------------------------------------
	const size_t ptr_size = 2;
	//create a dev ptr
	thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(ptr_size);
	//TODO: ugly hardcoded intial index again!
	dev_ptr[0] = 100;

	std::cout << "done allocating\n";
	//extract the raw pointer
	int * index_ptr = thrust::raw_pointer_cast(dev_ptr);

	//stopwatches
	const unsigned int interaction_watch = StopWatch::create();
	const unsigned int communication_watch = StopWatch::create();

	//track whether an interaction step was made
	thrust::device_vector<bool> normal(vector_size);

	// transfer the input to the device
	//---------------------------------
	thrust::device_vector<int> lhs_ids(host_lhs_ids), lhs_arities(host_lhs_arities), lhs_p1s(host_lhs_p1s), lhs_p2s(host_lhs_p2s), lhs_p3s(host_lhs_p3s), lhs_p4s(host_lhs_p4s);
	device_vector<name_type> lhs_names(host_lhs_names);
	thrust::device_vector<int> rhs_ids(host_rhs_ids), rhs_arities(host_rhs_arities), rhs_p1s(host_rhs_p1s), rhs_p2s(host_rhs_p2s), rhs_p3s(host_rhs_p3s), rhs_p4s(host_rhs_p4s);
	device_vector<name_type> rhs_names(host_rhs_names);


	// ---------------
	// input iterators
	//----------------
	//resize to full vector size
	lhs_ids.resize(vector_size); lhs_arities.resize(vector_size); lhs_names.resize(vector_size);
	lhs_p1s.resize(vector_size); lhs_p2s.resize(vector_size); lhs_p3s.resize(vector_size); lhs_p4s.resize(vector_size);
	rhs_ids.resize(vector_size); rhs_arities.resize(vector_size); rhs_names.resize(vector_size);
	rhs_p1s.resize(vector_size); rhs_p2s.resize(vector_size); rhs_p3s.resize(vector_size); rhs_p4s.resize(vector_size);

	//input
	agent_iter lhs_iter_begin = thrust::make_zip_iterator( 
		thrust::make_tuple( lhs_ids.begin(),lhs_names.begin(),lhs_arities.begin(),
		lhs_p1s.begin(),lhs_p2s.begin(),lhs_p3s.begin(), lhs_p4s.begin()) );
	auto rhs_iter_begin = thrust::make_zip_iterator(
		thrust::make_tuple( rhs_ids.begin(), rhs_names.begin(), rhs_arities.begin(),
		rhs_p1s.begin(), rhs_p2s.begin(), rhs_p3s.begin(), rhs_p4s.begin() ) );

	auto lhs_iter_end = thrust::make_zip_iterator(
		thrust::make_tuple( lhs_ids.end(),lhs_names.end(),lhs_arities.end(),
		lhs_p1s.end(),lhs_p2s.end(),lhs_p3s.end(),lhs_p4s.end()) );
	auto rhs_iter_end = thrust::make_zip_iterator(
		thrust::make_tuple( rhs_ids.end(), rhs_names.end(), rhs_arities.end(),
		rhs_p1s.end(), rhs_p2s.end(), rhs_p3s.end(), rhs_p4s.end() ) );

	equation_iter in_iter_begin = make_zip_iterator(make_tuple(lhs_iter_begin, rhs_iter_begin));
	auto in_iter_end = make_zip_iterator(make_tuple(lhs_iter_end, rhs_iter_end));
	//end of the actual input
	equation_iter input_end = in_iter_begin + input_size;

	//print_range("ids", lhs_names.begin(), lhs_names.end());
	//print_range("ids", rhs_names.begin(), rhs_names.end());

	// additional result vectors
	// TODO implement those

	//functor creation
	soa::ackermann_functor ack_functor(index_ptr);
	fibonacci_functor fib_functor(index_ptr);
	soa::algae_functor alg_functor(index_ptr);
	//add_functor functor;

	//loop variables
	int num_interactions = 0;
	int num_communications = 0;
	int num_loops = 0;
	int total_interactions = 0;
	
	if (verbose) {
		std::cout << "input:\n---------------\n";
		print_equations( host_vector<equation>(in_iter_begin, in_iter_end) );
	}

	//auxiliary iterators for comm step
		//auxiliary vector tuple to avoid in-place reduce
	//lhs
	thrust::device_vector<int> aux_lhs_ids(aux_input_size), 
		aux_lhs_arities(aux_input_size),
		aux_lhs_p1s(aux_input_size),
		aux_lhs_p2s(aux_input_size),
		aux_lhs_p3s(aux_input_size),
		aux_lhs_p4s(aux_input_size);
	device_vector<name_type> aux_lhs_names(aux_input_size);
	//rhs
	thrust::device_vector<int> aux_rhs_ids(aux_input_size), 
		aux_rhs_arities(aux_input_size),
		aux_rhs_p1s(aux_input_size),
		aux_rhs_p2s(aux_input_size),
		aux_rhs_p3s(aux_input_size),
		aux_rhs_p4s(aux_input_size);
	device_vector<name_type> aux_rhs_names(aux_input_size);
	// auxiliary iterators
	// aux iter lhs
	agent_iter aux_lhs_iter_begin = thrust::make_zip_iterator( 
		thrust::make_tuple( aux_lhs_ids.begin(),aux_lhs_names.begin(),aux_lhs_arities.begin(),
		aux_lhs_p1s.begin(),aux_lhs_p2s.begin(),aux_lhs_p3s.begin(), aux_lhs_p4s.begin()) );
	auto aux_rhs_iter_begin = thrust::make_zip_iterator(
		thrust::make_tuple( aux_rhs_ids.begin(), aux_rhs_names.begin(), aux_rhs_arities.begin(),
		aux_rhs_p1s.begin(), aux_rhs_p2s.begin(), aux_rhs_p3s.begin(), aux_rhs_p4s.begin() ) );
	// aux iter rhs
	auto aux_lhs_iter_end = thrust::make_zip_iterator(
		thrust::make_tuple( aux_lhs_ids.end(),aux_lhs_names.end(),aux_lhs_arities.end(),
		aux_lhs_p1s.end(),aux_lhs_p2s.end(),aux_lhs_p3s.end(),aux_lhs_p4s.end()) );
	auto aux_rhs_iter_end = thrust::make_zip_iterator(
		thrust::make_tuple( aux_rhs_ids.end(), aux_rhs_names.end(), aux_rhs_arities.end(),
		aux_rhs_p1s.end(), aux_rhs_p2s.end(), aux_rhs_p3s.end(), aux_rhs_p4s.end() ) );
	equation_iter aux_iter_begin = make_zip_iterator(make_tuple(aux_lhs_iter_begin, aux_rhs_iter_begin));
	equation_iter aux_iter_end = make_zip_iterator(make_tuple(aux_lhs_iter_end, aux_rhs_iter_end));
	
		
	//additional result arrays
	// -----------------------
	//// A
	//thrust::device_vector<int> a_lhs_ids(input_size, -1), a_lhs_arities(input_size), a_lhs_p1s(input_size), a_lhs_p2s(input_size), a_lhs_p3s(input_size), a_lhs_p4s(input_size);
	//device_vector<name_type> a_lhs_names(input_size, 'Z');
	//thrust::device_vector<int> a_rhs_ids(input_size, -1), a_rhs_arities(input_size), a_rhs_p1s(input_size), a_rhs_p2s(input_size), a_rhs_p3s(input_size), a_rhs_p4s(input_size);
	//device_vector<name_type> a_rhs_names(input_size, 'Z');
	//// B
	//thrust::device_vector<int> b_rhs_ids(input_size, -1), b_rhs_arities(input_size), b_rhs_p1s(input_size), b_rhs_p2s(input_size), b_rhs_p3s(input_size), b_rhs_p4s(input_size);
	//device_vector<name_type> b_rhs_names(input_size, 'Z');
	//thrust::device_vector<int> b_lhs_ids(input_size, -1), b_lhs_arities(input_size), b_lhs_p1s(input_size), b_lhs_p2s(input_size), b_lhs_p3s(input_size), b_lhs_p4s(input_size);
	//device_vector<name_type> b_lhs_names(input_size, 'Z');
	//// C
	//thrust::device_vector<int> c_lhs_ids(input_size, -1), c_lhs_arities(input_size), c_lhs_p1s(input_size), c_lhs_p2s(input_size), c_lhs_p3s(input_size), c_lhs_p4s(input_size);
	//device_vector<name_type> c_lhs_names(input_size, 'Z');
	//thrust::device_vector<int> c_rhs_ids(input_size, -1), c_rhs_arities(input_size), c_rhs_p1s(input_size), c_rhs_p2s(input_size), c_rhs_p3s(input_size), c_rhs_p4s(input_size);
	//device_vector<name_type> c_rhs_names(input_size, 'Z');

	//equation_iter c_iter_begin = input_end + input_size * 2;
	//auto c_iter_end = input_end + input_size * 3;
	

	/*--------------------------
	// START OF INTERACTION LOOP
	----------------------------*/
	do {

	//virtual result array A
	agent_iter a_lhs_iter_begin = lhs_iter_begin + input_size;
	agent_iter a_rhs_iter_begin = rhs_iter_begin + input_size;
	agent_iter a_lhs_iter_end = lhs_iter_begin + input_size * 2;
	agent_iter a_rhs_iter_end = rhs_iter_begin + input_size * 2;
	equation_iter a_iter_begin = make_zip_iterator(make_tuple(a_lhs_iter_begin, a_rhs_iter_begin));
	auto a_iter_end = make_zip_iterator(make_tuple(a_lhs_iter_end, a_rhs_iter_end));

	//virtual result array B
	agent_iter b_lhs_iter_begin = lhs_iter_begin + input_size * 2;
	agent_iter b_rhs_iter_begin = rhs_iter_begin + input_size * 2;
	agent_iter b_lhs_iter_end = lhs_iter_begin + input_size * 3;
	agent_iter b_rhs_iter_end = rhs_iter_begin + input_size * 3;
	equation_iter b_iter_begin = make_zip_iterator(make_tuple(b_lhs_iter_begin, b_rhs_iter_begin));
	auto b_iter_end = make_zip_iterator(make_tuple(b_lhs_iter_end, b_rhs_iter_end));

	//virtual result array C
	agent_iter c_lhs_iter_begin = lhs_iter_begin + input_size * 3;
	agent_iter c_rhs_iter_begin = rhs_iter_begin + input_size * 3;
	agent_iter c_lhs_iter_end = lhs_iter_begin + input_size * 4;
	agent_iter c_rhs_iter_end = rhs_iter_begin + input_size * 4;
	equation_iter c_iter_begin = make_zip_iterator(make_tuple(c_lhs_iter_begin, c_rhs_iter_begin));
	auto c_iter_end = make_zip_iterator(make_tuple(c_lhs_iter_end, c_rhs_iter_end));

	//// invalid test
	//if (thrust::any_of(in_iter_begin, in_iter_end, is_invalid_tuple() ) )  {
	//	std::cout << "invalid tuple before int step detected!\n";
	//	break;
	//}

		
	if (verbose) {
		std::cout << "input:\n---------------\n";
		print_equations( host_vector<equation>(in_iter_begin, input_end) );
	}

	//-----------------
	// interaction step
	// ----------------
	//virtual result vectors are at input_size * {1,2,3}
	switch( ruleset ) {
	case 1:
		thrust::for_each(
			make_zip_iterator(make_tuple(lhs_iter_begin, rhs_iter_begin, a_lhs_iter_begin, a_rhs_iter_begin,
					b_lhs_iter_begin, b_rhs_iter_begin, c_lhs_iter_begin, c_rhs_iter_begin, normal.begin())),
			make_zip_iterator(make_tuple(lhs_iter_begin + input_size, rhs_iter_begin + input_size, a_lhs_iter_end, a_rhs_iter_end,
					b_lhs_iter_end, b_rhs_iter_end, c_lhs_iter_end, c_rhs_iter_end, normal.end())),
			ack_functor
		);
		break;
	case 2:
		thrust::for_each(
			make_zip_iterator(make_tuple(lhs_iter_begin, rhs_iter_begin, 
										lhs_iter_begin + input_size, rhs_iter_begin + input_size,
										lhs_iter_begin + input_size * 2, rhs_iter_begin + input_size * 2,
										lhs_iter_begin + input_size * 3, rhs_iter_begin + input_size * 3,
										normal.begin())),
			make_zip_iterator(make_tuple(lhs_iter_begin + input_size, rhs_iter_begin + input_size, 
										lhs_iter_begin + input_size * 2, rhs_iter_begin + input_size * 2,
										lhs_iter_begin + input_size * 3, rhs_iter_begin + input_size * 3,
										lhs_iter_begin + input_size * 4, rhs_iter_begin + input_size * 4,
										normal.end())),
			alg_functor
		);
		break;
	case 3:
		thrust::for_each(
			make_zip_iterator(make_tuple(lhs_iter_begin, rhs_iter_begin, 
										lhs_iter_begin + input_size, rhs_iter_begin + input_size,
										lhs_iter_begin + input_size * 2, rhs_iter_begin + input_size * 2,
										lhs_iter_begin + input_size * 3, rhs_iter_begin + input_size * 3,
										normal.begin())),
			make_zip_iterator(make_tuple(lhs_iter_begin + input_size, rhs_iter_begin + input_size, 
										lhs_iter_begin + input_size * 2, rhs_iter_begin + input_size * 2,
										lhs_iter_begin + input_size * 3, rhs_iter_begin + input_size * 3,
										lhs_iter_begin + input_size * 4, rhs_iter_begin + input_size * 4,
										normal.end())),
			fib_functor
		);
		break;
	}
	// interaction step done
	// ---------------------

	//raw interaction kernel measure time
	//StopWatch::get(interaction_watch).stop();

	//StopWatch::get(interaction_watch).start();
	//check if we performed one or more interactions
	num_interactions = thrust::count(normal.begin(), normal.end(), true);
	if (num_interactions > 0) { 
		num_loops++;
		if (verbose)
		std::cout << "interactions: " << num_interactions << "\n"; 
	}
	else {
		if (verbose) {
			std::cout << "equations are in int normal form\n\n";
		}
		break;
	}
	
	//print_range("ids", lhs_names.begin(), lhs_names.end());
	if (verbose) {
		std::cout << "result of int step:\n----------------\n";
		print_equations( host_vector<equation>(in_iter_begin, input_end) );
		print_equations( host_vector<equation>(a_iter_begin, a_iter_end) );
		print_equations( host_vector<equation>(b_iter_begin, b_iter_end) );
		print_equations( host_vector<equation>(c_iter_begin, c_iter_end) );
	}

	//dummy removal
	//TODO use reduce by key for speedup
	//input end is set to the end of the non-dummies
	//in_iter_end remains at the actual end of the vectors
	input_end = thrust::remove_if(in_iter_begin, in_iter_end, is_dummy_equation_functor());
	input_size = input_end - in_iter_begin;
	//cleanup functor
	//thrust::fill(lhs_ids.begin() + input_size, lhs_ids.end(), 0);
	//// invalid test
	//if (thrust::any_of(in_iter_begin, in_iter_end, is_invalid_tuple() ) )  {
	//	std::cout << "invalid tuple after merge detected!\n";
	//	break;
	//}
	
	if (verbose) {
		std::cout << "post merge:\n--------------\n";
		print_equations( host_vector<equation>(in_iter_begin, input_end) );
		print_equations( host_vector<equation>(input_end, in_iter_end) );
	}


	//StopWatch::get(communication_watch).start();
	
	StopWatch::get(interaction_watch).start();
	// sorting and aux vector handling
	// -------------------------------
	//resize aux vectors if necessary
	if (input_size > aux_input_size) {
		aux_input_size = input_size;
		//resize and set iterators
		aux_lhs_ids.resize(aux_input_size); aux_lhs_arities.resize(aux_input_size); aux_lhs_names.resize(aux_input_size);
		aux_lhs_p1s.resize(aux_input_size); aux_lhs_p2s.resize(aux_input_size); aux_lhs_p3s.resize(aux_input_size); aux_lhs_p4s.resize(aux_input_size);
		aux_rhs_ids.resize(aux_input_size); aux_rhs_arities.resize(aux_input_size); aux_rhs_names.resize(aux_input_size);
		aux_rhs_p1s.resize(aux_input_size); aux_rhs_p2s.resize(aux_input_size); aux_rhs_p3s.resize(aux_input_size); aux_rhs_p4s.resize(aux_input_size);
		
	}
	//set iterators
	// aux iter lhs
	aux_rhs_iter_begin = thrust::make_zip_iterator(
		thrust::make_tuple( aux_rhs_ids.begin(), aux_rhs_names.begin(), aux_rhs_arities.begin(),
		aux_rhs_p1s.begin(), aux_rhs_p2s.begin(), aux_rhs_p3s.begin(), aux_rhs_p4s.begin() ) );
	aux_lhs_iter_begin = thrust::make_zip_iterator( 
		thrust::make_tuple( aux_lhs_ids.begin(),aux_lhs_names.begin(),aux_lhs_arities.begin(),
		aux_lhs_p1s.begin(),aux_lhs_p2s.begin(),aux_lhs_p3s.begin(), aux_lhs_p4s.begin()) );

	equation_iter aux_iter_begin = make_zip_iterator(make_tuple(aux_lhs_iter_begin, aux_rhs_iter_begin));

	//get get a vector for the permutations
	//thrust::device_vector<int> permutation(input_size);
	//thrust::sequence(permutation.begin(), permutation.end());

	//sort by id of the lhs
	//thrust::sort_by_key(lhs_ids.begin(), lhs_ids.begin() + input_size, permutation.begin());

	//orient sorting
	//thrust::sort(in_iter_begin, input_end, soa::comm_equation_greater());

	//if (verbose) {
	//	print_range("permutation", permutation.begin(), permutation.end()); 
	//}
	//if (verbose) {
	//	std::cout << "empty aux vector (size = " << aux_input_size <<" ):\n--------------\n";
	//	print_equations( host_vector<equation>(aux_iter_begin, aux_iter_begin + aux_input_size) );
	//}
	////gather the sorted elements into the aux vector
	//thrust::gather(permutation.begin(), permutation.end(), in_iter_begin, aux_iter_begin);

	//crappy sort
	// TODO copying isnt really slow
	thrust::copy(in_iter_begin, input_end, aux_iter_begin);
	thrust::sort_by_key(lhs_ids.begin(), lhs_ids.begin() + input_size, aux_iter_begin);
	
	if (verbose) {
		std::cout << "sorted aux vector (size = " << aux_input_size <<" ):\n--------------\n";
		print_equations( host_vector<equation>(aux_iter_begin, aux_iter_begin + aux_input_size) );
	}
	StopWatch::get(interaction_watch).stop();
	
	/*------------------------
	/ Communication step
	--------------------------*/
	StopWatch::get(communication_watch).start();
	// parallel communication
	num_communications = comm_step_parallel(
											aux_iter_begin, aux_iter_begin + input_size, 
											in_iter_begin, input_end,
											aux_lhs_ids.begin(), aux_lhs_ids.begin() + input_size,
											input_size,
											communication_watch);
	input_size = input_end - in_iter_begin;

	if (verbose) {
		std::cout << "post communicate (input_size = " << input_size <<" ):\n--------------\n";
		print_equations( host_vector<equation>(in_iter_begin, input_end) );
		print_equations( host_vector<equation>(input_end, in_iter_end) );
	}
	
	//// invalid test
	//if (thrust::any_of(in_iter_begin, in_iter_end, is_invalid_tuple() ) )  {
	//	std::cout << "invalid tuple after communicate detected!\n";
	//	break;
	//}
	/*------------------------------
	/ communication end
	-------------------------------*/
	StopWatch::get(communication_watch).stop();

	
	/*------------------------------
	/ cleanup
	-------------------------------*/
	//interaction statisitics
	total_interactions += num_interactions;

	// exit condition
	// no interactions and no communications in the previous loop
	if (num_interactions < 1 && num_communications < 1) {
		break;
	}

	//resize if necessary
	if (vector_size < (input_size * 4) ) {
		//resize
		vector_size = input_size * 4;
		//resize to full vector size
		lhs_ids.resize(vector_size); lhs_arities.resize(vector_size); lhs_names.resize(vector_size);
		lhs_p1s.resize(vector_size); lhs_p2s.resize(vector_size); lhs_p3s.resize(vector_size); lhs_p4s.resize(vector_size);
		rhs_ids.resize(vector_size); rhs_arities.resize(vector_size); rhs_names.resize(vector_size);
		rhs_p1s.resize(vector_size); rhs_p2s.resize(vector_size); rhs_p3s.resize(vector_size); rhs_p4s.resize(vector_size);
		//resize normal vector
		normal.resize(vector_size);
		thrust::fill(normal.begin(), normal.end(), false);
		//redo iterators
		//input
		lhs_iter_begin = thrust::make_zip_iterator( 
			thrust::make_tuple( lhs_ids.begin(),lhs_names.begin(),lhs_arities.begin(),
			lhs_p1s.begin(),lhs_p2s.begin(),lhs_p3s.begin(), lhs_p4s.begin()) );
		rhs_iter_begin = thrust::make_zip_iterator(
			thrust::make_tuple( rhs_ids.begin(), rhs_names.begin(), rhs_arities.begin(),
			rhs_p1s.begin(), rhs_p2s.begin(), rhs_p3s.begin(), rhs_p4s.begin() ) );

		lhs_iter_end = thrust::make_zip_iterator(
			thrust::make_tuple( lhs_ids.end(),lhs_names.end(),lhs_arities.end(),
			lhs_p1s.end(),lhs_p2s.end(),lhs_p3s.end(),lhs_p4s.end()) );
		rhs_iter_end = thrust::make_zip_iterator(
			thrust::make_tuple( rhs_ids.end(), rhs_names.end(), rhs_arities.end(),
			rhs_p1s.end(), rhs_p2s.end(), rhs_p3s.end(), rhs_p4s.end() ) );

		in_iter_begin = make_zip_iterator(make_tuple(lhs_iter_begin, rhs_iter_begin));
		in_iter_end = make_zip_iterator(make_tuple(lhs_iter_end, rhs_iter_end));
		//end of the actual input
		input_end = in_iter_begin + input_size;

		if (verbose) {
		std::cout << "post resize (input_size = " << input_size <<" )::\n--------------\n";
			print_equations( host_vector<equation>(in_iter_begin, input_end) );
			print_equations( host_vector<equation>(input_end, in_iter_end) );
		}
	}

	//cleanup functor
	thrust::fill(lhs_ids.begin() + input_size, lhs_ids.end(), 0);
	thrust::fill(rhs_ids.begin() + input_size, rhs_ids.end(), 0);

	//if (verbose) {
	//std::cout << "post cleanup:\n--------------\n";
	//	print_equations( host_vector<equation>(in_iter_begin, input_end) );
	//	print_equations( host_vector<equation>(input_end, in_iter_end) );
	//}

	
	normal.resize(input_size, false);

	if (verbose) {
		std::cout << "\n-------------------------------------\n";
		std::cout << "end of interaction loop\n";
		std::cout << "-------------------------------------\n";
		if (pause) {
			std::cout << "press a key to continue\n";
			getch();
		}
	}
	} while (true);
	/*---------------------
	/ END OF LOOP
	-----------------------*/

	if (print_final) {
		std::cout << "\n result\n-----------\n";
		print_equations( host_vector<equation>(in_iter_begin, input_end) );
	}
	
	//compute result for addition tests
	int s = thrust::count_if(in_iter_begin, input_end, has_name_tuple('S'));
	//std::cout << "S symbols: " << s << std::endl;
	
	//highest index:
	std::cout << "maximum index used: " << dev_ptr[0] << std::endl;
	// stop and destroy timer
	float int_time = StopWatch::get(interaction_watch).getTime() / 1000.0;
	float com_time = StopWatch::get(communication_watch).getTime() / 1000.0;
	printf ("total interaction time %.5f seconds.\n", int_time );
	printf ("total communication time %.5f seconds.\n", com_time );
	StopWatch::destroy(communication_watch);
	StopWatch::destroy(interaction_watch);

	return result_info(s, num_loops, total_interactions, com_time, int_time);
	
}


//
//// interaction combinators
//struct add_functor
//{
//template <typename Tuple>
//__host__ __device__
//void operator()(Tuple equations)
//{
//	//get lhs and rhs of the equations
//	agent lhs = thrust::get<0>(equations);
//	agent rhs = thrust::get<1>(equations);
//
//	// default: no interaction was performed
//	thrust::get<8>(equations) = false;
//
//		
//	// if both sides are agents
//	// rewrite the equations
//	
//	if (is_upper( get<NAME>(lhs) ) && is_upper( get<NAME>(lhs) )) {
//
//		//sort pair lexicographically (active pairs are symmetric)
//		if (get<NAME>(rhs) < get<NAME>(lhs) ) {
//			lhs = rhs;
//			rhs = thrust::get<0>(equations);
//		}
//		
//		name_type lhs_name = get<NAME>(lhs);
//		name_type rhs_name = get<NAME>(rhs);
//
//		// A(r, y) >< 0 => r~y
//		if (lhs_name == 'A' && rhs_name == 'O') {
//	
//			//r~y
//			thrust::get<0>(equations) = make_agent(get<P1>(lhs), 'r', 0);
//			thrust::get<1>(equations) = make_agent(get<P2>(lhs), 'y', 0);
//
//			thrust::get<8>(equations) = true; // we performed an interaction
//		}
//		// A(r, y) >< S(x) => r~S(w), A(w, y)~x
//		else if (lhs_name == 'A' && rhs_name == 'S') {
//			int new_index = thrust::get<9>(equations);
//			thrust::get<9>(equations) = -1; //remove used index
//			//wires
//			int x = get<P1>(rhs);
//			int r = get<P1>(lhs);
//			int w = new_index; // new variable w
//	
//			// r~S(w)
//			get<P1>(rhs) = w;
//			thrust::get<0>(equations) = make_agent(r, 'r', 0);
//			thrust::get<1>(equations) = rhs;
//			// A(w, y)~x
//			get<P1>(lhs) = w;
//			thrust::get<2>(equations) = make_agent(x, 'x', 0);
//			thrust::get<3>(equations) = lhs;
//
//			thrust::get<8>(equations) = true; // we performed an interaction
//		}
//	}
//	}
//};
