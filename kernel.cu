
#include "cuda.h"
#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
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
#include <thrust/device_malloc.h>
#include <thrust\device_free.h>
#include <thrust\device_ptr.h>
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
#include <thrust/set_operations.h>
#include <thrust\merge.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <cutil_inline.h>
#include <cutil.h>
#include "agent.cuh"
#include "ackermann.h"
#include "communication.h"
#include "fibonacci.h"
#include "algae.h"
#include "config_manager.h"

#include <stopwatch.h>
#include <stopwatch_win.h>
//#include "parse_net.h"

//print a vector
//template <typename Iterator>
//void print_range(const std::string& name, Iterator first, Iterator last)
//{
//	typedef typename std::iterator_traits<Iterator>::value_type T;
//
//	std::cout << name << ": ";
//	thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
//	std::cout << std::endl;
//}

/*-------------------------
Interaction Nets Evaluation
-------------------------*/

result_info partition_interaction_step(const int ruleset, 
	thrust::host_vector<Agent>& in_lhs_host, thrust::host_vector<Agent>& in_rhs_host, const bool verbose, const bool print_final_result, const bool pause)
{
	// ---------
	// constants
	// ---------
	//input size
	int input_size = in_lhs_host.size();

	//make a device pointer for tracking agent indices
	//------------------------------------------
	const size_t ptr_size = 1;
	//create a dev ptr
	thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(ptr_size);
	//TODO: ugly hardcoded intial index again!
	dev_ptr[0] = 100;
	int maximum_index = 100; //track max index, to check for int overflow

	std::cout << "done allocating\n";
	//extract the raw pointer
	int * index_ptr = thrust::raw_pointer_cast(dev_ptr);


	// transfer the input to the device
	//---------------------------------
	thrust::device_vector<Agent> in_lhs(in_lhs_host);
	thrust::device_vector<Agent> in_rhs(in_rhs_host);

	// paritioning
	//------------

	//zip lhs and rhs together
	equation_iter in_begin = thrust::make_zip_iterator( thrust::make_tuple(in_lhs.begin(), in_rhs.begin() ) );
	equation_iter in_end = thrust::make_zip_iterator( thrust::make_tuple(in_lhs.end(), in_rhs.end() ) );

	//get the partition between aps and tlvs
	is_active_functor active_functor;
	equation_iter in_partition = thrust::partition(in_begin, in_end, active_functor);
	
	int ap_size = in_partition - in_begin;
	//create two vector pairs: active pairs (aps) and top-level variable equations (tlvs)
	thrust::device_vector<Agent> aps_L(in_lhs.begin(), in_lhs.begin() + ap_size);
	thrust::device_vector<Agent> aps_R(in_rhs.begin(), in_rhs.begin() + ap_size);
	thrust::device_vector<Agent> tlvs_L(in_lhs.begin() + ap_size, in_lhs.end());
	thrust::device_vector<Agent> tlvs_R(in_rhs.begin() + ap_size, in_rhs.end());
	
	equation_iter aps_begin = thrust::make_zip_iterator( thrust::make_tuple(aps_L.begin(), aps_R.begin() ) );
	equation_iter aps_end = thrust::make_zip_iterator( thrust::make_tuple(aps_L.end(), aps_R.end() ) );
	equation_iter tlvs_begin = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.begin(), tlvs_R.begin() ) );
	equation_iter tlvs_end = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.end(), tlvs_R.end() ) );

	std::cout << "aps:\n";
	//TODO sorting aps is perhaps unneccessary
	thrust::sort(aps_begin, aps_end, smaller_tlv_functor());
	print_equations(aps_L, aps_R);
	thrust::sort(tlvs_begin, tlvs_end, smaller_tlv_functor());
	std::cout << "tlvs:\n";
	print_equations(tlvs_L, tlvs_R);

	//new result arrays
	thrust::device_vector<Agent> a_L(ap_size);
	thrust::device_vector<Agent> a_R(ap_size);
	thrust::device_vector<Agent> b_L(ap_size);
	thrust::device_vector<Agent> b_R(ap_size);
	thrust::device_vector<Agent> c_L(ap_size);
	thrust::device_vector<Agent> c_R(ap_size);
	
	//result vectors
	thrust::host_vector<Agent> result_lhs(in_lhs.size());
	thrust::host_vector<Agent> result_rhs(in_rhs.size());

	if (verbose) {
		print_equations(in_lhs, in_rhs);
	}

	//loop variable
	//int num_interactions = 0;
	int num_loops = 0;
	//int total_interactions = 0;
	int total_interactions_partition = 0;

	//functor selection
	//-----------------
	ackermann_functor ack_functor(index_ptr);
	//add_functor functor;
	//ic_functor functor2;
	//fibonacci_functor functor(index_ptr);
	algae_functor alg_functor(index_ptr);

	//stopwatches
	const unsigned int interaction_watch = StopWatch::create();
	const unsigned int communication_watch = StopWatch::create();

	// ---------------
	// start of loop
	// ---------------
	do {

	if (verbose) {
		std::cout << "aps:\n";
		print_equations(aps_begin, aps_end);
	}

	//original int start position
	StopWatch::get(interaction_watch).start();
	// -------------------------------------------
	// apply interaction step to list of equations
	// -------------------------------------------
	switch (ruleset) {
	case 1:
		//ap only version
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(
				aps_L.begin(),
				aps_R.begin(),
				a_L.begin(), a_R.begin(), b_L.begin(), b_R.begin(), c_L.begin(), c_R.begin(), thrust::make_discard_iterator() ) ),
			thrust::make_zip_iterator(thrust::make_tuple(
				aps_L.end(),
				aps_R.end(),
				a_L.end(),   a_R.end(),   b_L.end(),   b_R.end(),   c_L.end(),   c_R.end(),   thrust::make_discard_iterator()  ) ),
			ack_functor
		);
		break;
	case 2:
		//ap only version
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(
				aps_L.begin(),
				aps_R.begin(),
				a_L.begin(), a_R.begin(), b_L.begin(), b_R.begin(), c_L.begin(), c_R.begin(), thrust::make_discard_iterator() ) ),
			thrust::make_zip_iterator(thrust::make_tuple(
				aps_L.end(),
				aps_R.end(),
				a_L.end(),   a_R.end(),   b_L.end(),   b_R.end(),   c_L.end(),   c_R.end(),   thrust::make_discard_iterator()  ) ),
			alg_functor
		);
		break;

	}
	
	//------------------------------------------------
	// end of interaction step
	//------------------------------------------------
	
	//StopWatch::get(interaction_watch).stop();

	if (verbose) {
		std::cout << "partition interactions: " << ap_size << "\n"; 
	}

	// partition interaction count
	total_interactions_partition += ap_size;

	//StopWatch::get(interaction_watch).stop();
	// merge equation arrays
	// TODO less copy operations
	// AP only copying
	// ----------------

	aps_L.resize(ap_size * 4);
	aps_R.resize(ap_size * 4);

	// append A to aps
	thrust::copy(a_L.begin(), a_L.end(), aps_L.begin() + ap_size);
	thrust::copy(a_R.begin(), a_R.end(), aps_R.begin() + ap_size);

	// append B to aps
	thrust::copy(b_L.begin(), b_L.end(), aps_L.begin() + ap_size*2);
	thrust::copy(b_R.begin(), b_R.end(), aps_R.begin() + ap_size*2);
	
	// append C to aps
	thrust::copy(c_L.begin(), c_L.end(), aps_L.begin() + ap_size*3);
	thrust::copy(c_R.begin(), c_R.end(), aps_R.begin() + ap_size*3);

	if (verbose) {
		std::cout << "aps after result merge:\n---------------\n";
		print_equations(aps_L, aps_R);
	}

	//remove dummy equations for aps
	// TODO do I need to reset aps_begin aswell?
	aps_begin = thrust::make_zip_iterator( thrust::make_tuple(aps_L.begin(), aps_R.begin() ) );
	aps_end = thrust::make_zip_iterator( thrust::make_tuple(aps_L.end(), aps_R.end() ) );
	// note: remove_if is stable
	equation_iter aps_new_end = thrust::remove_if(
		aps_begin,
		aps_end,
		dummy_test_functor_tuple()
		);
	
	//TODO make this conditional if no dummies have been removed
	ap_size = aps_new_end - aps_begin;
	aps_L.resize(ap_size);
	aps_R.resize(ap_size);
	//update iterators
	aps_begin = thrust::make_zip_iterator( thrust::make_tuple(aps_L.begin(), aps_R.begin() ) );
	aps_end = thrust::make_zip_iterator( thrust::make_tuple(aps_L.end(), aps_R.end() ) );
	
	//if (verbose) {
	//	std::cout << "aps after dummy removal:\n---------------\n";
	//	print_equations(aps_L, aps_R);
	//}

	//aps
	//partition result again by aps and tlvs
	//TODO is it faster just to sort?
	equation_iter aps_partition = thrust::partition(aps_begin, aps_end, active_functor);
	if (verbose) {
		std::cout << "new tlvs: " << (aps_end - aps_partition) << std::endl;
	}

	//we only care about sortedness of tlvs
	thrust::sort(aps_partition, aps_end, smaller_tlv_functor());
	
	if (verbose) {
		std::cout << "aps after dummy removal and sort:\n---------------\n";
		print_equations(aps_L, aps_R);
	}

	//original int watch position
	StopWatch::get(interaction_watch).stop();

	// --------------------------------------------------------------------
	// communication part: find eligible tlv pairs and reduce them to an AP
	// --------------------------------------------------------------------
	/*StopWatch::get(communication_watch).start();*/

	// the result may contain communication-eligible pairs of TLVs
	// sort out duplicate tlvs with unique
	// ------------------------------------
	//non-uniques vector
	thrust::device_vector<equation> uniques(aps_L.size());
	//number of APs after interaction
	int num_post_int_aps = aps_partition - aps_begin;
	//number of TLVS after interaction
	int num_post_int_tlvs = aps_end - aps_partition;
	//get unique TLV equations
	auto uniques_end = thrust::unique_copy(aps_partition, aps_end, uniques.begin(), equal_tlv_functor());
	int num_uniques = uniques_end - uniques.begin();
	if (num_uniques < num_post_int_tlvs) {
		// there are non_unique tlvs, we need to move them from the aps vector to the tlvs vector
		uniques.resize(num_uniques);
		if (verbose) {
		std::cout << "we have uniques:" << num_uniques << " (of " << num_post_int_tlvs << " tlvs), loop "<< num_loops <<"   \n---------------------\n";
		print_equations(uniques);
		}
		//get the non_uniques
		//TODO maximum size for non-unique tlvs: num_post_int_tlvs - num_uniques?
		thrust::device_vector<equation> non_uniques(num_post_int_tlvs - num_uniques);
		//get the non_uniques via set_difference
		//unique_copy is not stable, so sort it again
		thrust::sort(uniques.begin(), uniques.end(), smaller_tlv_functor());	
		thrust::sort(aps_partition, aps_end, smaller_tlv_functor());

		if (verbose) {
			std::cout << "aps, tlv partition, loop " << num_loops << ": \n";
			print_equations(aps_partition, aps_end);
		}
		assert( ((aps_end - aps_partition) - num_uniques) == num_post_int_tlvs - uniques.size());
		//random weirdness testing
		//if (num_loops == 16) {
		//	thrust::device_vector<equation> bla(aps_partition + config_manager::get_int("set_diff_test_start"), aps_partition + config_manager::get_int("set_diff_test_length"));
		//	std::cout << "auxiliary aps testing " << num_loops << ": \n";
		//	print_equations(bla);
		//	auto non_uniques_end = thrust::set_difference(bla.begin(), bla.end(), uniques.begin(), uniques.end(), non_uniques.begin(), smaller_tlv_functor());
		//	std::cout << "set_diff with aps empty:\n";
		//	int size = non_uniques_end - non_uniques.begin();
		//	print_equations(non_uniques);
		//}
		// set_difference is fine, no in-place update
		auto non_uniques_end = thrust::set_difference(aps_partition, aps_end, uniques.begin(), uniques.end(), non_uniques.begin(), smaller_tlv_functor());
		if (verbose) {
			std::cout << "got past set difference\n";
		}
		//if (verbose) {
		//	std::cout << "non_uniques pre _resize: \n---------------------\n";
		//	print_equations(non_uniques);
		//}
		int num_non_uniques = non_uniques_end - non_uniques.begin();
		if (verbose) {
			std::cout << "num_non_uniques: " << num_non_uniques << std::endl;
		}
		non_uniques.resize(num_non_uniques);
		if (verbose) {
			std::cout << "non_uniques:" << num_non_uniques << " \n---------------------\n";
			print_equations(non_uniques);
		}
		// the non_uniques vector should be completely filled, assert that
		//std::cout << (non_uniques_end - non_uniques.begin()) << " vs " << (non_uniques.end() - non_uniques.begin()) << std::endl;
		//assert( (non_uniques_end - non_uniques.begin()) == non_uniques.size() );
		//remove non_uniques from aps vector
		//TODO safe to assign aps_end right away?
		aps_end = thrust::unique(aps_partition, aps_end, equal_tlv_functor());
		//resize and update iterators
		aps_L.resize(aps_end - aps_begin);
		aps_R.resize(aps_end - aps_begin);
		aps_begin = thrust::make_zip_iterator( thrust::make_tuple(aps_L.begin(), aps_R.begin() ) );
		aps_end = thrust::make_zip_iterator( thrust::make_tuple(aps_L.end(), aps_R.end() ) );
		aps_partition = aps_begin + num_post_int_aps;
		if (verbose) {
			std::cout << "aps after unique removal:" << num_non_uniques << " \n---------------------\n";
			print_equations(aps_L, aps_R);
		}
		if (verbose) {
			std::cout << "previous tlvs: \n---------------------\n";
			print_equations(tlvs_L, tlvs_R);
		}
		//merge non_uniques into tlvs vector
		int old_tlv_size = tlvs_L.size();
		tlvs_L.resize(tlvs_L.size() + non_uniques.size());
		tlvs_R.resize(tlvs_R.size() + non_uniques.size());
		//tlv iterator update
		tlvs_begin = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.begin(), tlvs_R.begin() ) );
		tlvs_end = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.end(), tlvs_R.end() ) );
		// TODO remove in-place update
		//tlvs_end = thrust::merge(non_uniques.begin(), non_uniques.end(), tlvs_begin, tlvs_begin + old_tlv_size, tlvs_begin, smaller_tlv_functor());
		//instead of merge, copy and then sort
		//TODO triangle swap with extra vector and merge could be faster
		tlvs_end = thrust::copy(non_uniques.begin(), non_uniques.end(), tlvs_begin + old_tlv_size);
		thrust::sort( tlvs_begin, tlvs_end, smaller_tlv_functor() );
		if (verbose) {
			std::cout << "unique merged tlvs: \n---------------------\n";
			print_equations(tlvs_L, tlvs_R);
		}

	}

	//determine the maximum number of communications
	int max_communications = aps_end - aps_partition;
	int num_non_intersecting_tlvs = max_communications;

	thrust::device_vector<equation> tlv_intersection_1(0);
	// intersection 1: find the PREVIOUS tlvs that appear in the result
	// ----------------------------------------------------------------
	if ( tlvs_begin != tlvs_end && (max_communications > 0) ) {
		tlv_intersection_1.resize( (tlvs_end - tlvs_begin) + (aps_end - aps_partition) );
		thrust::sort(aps_partition, aps_end, smaller_tlv_functor());
		thrust::sort(tlvs_begin, tlvs_end, smaller_tlv_functor());
		if (verbose) {
			//std::cout << "tlvs:\n---------------\n";
			std::cout << "max communications: " << max_communications << std::endl;
			//std::cout << "got till 1nd intersection computation (sorted stuff before)\n";
		}
		//set-intersection is fine here, no in-place update
		auto intersection_end = thrust::set_intersection(tlvs_begin, tlvs_end, aps_partition, aps_end, tlv_intersection_1.begin(), smaller_tlv_functor());
		tlv_intersection_1.resize( intersection_end - tlv_intersection_1.begin() );
	}

	if (verbose) {
		//std::cout << "got till 2nd intersection computation\n";
	}

	// intersection 2: find the RESULT tlvs that appear in intersection 1
	// ------------------------------------------------------------------
	thrust::device_vector<equation> tlv_intersection_2(tlv_intersection_1.size());
	if (tlv_intersection_1.size() > 0 ) {
		//intersection is fine, no in-place update
		auto intersection_end2 = thrust::set_intersection(aps_partition, aps_end, tlv_intersection_1.begin(), tlv_intersection_1.end(), tlv_intersection_2.begin(), smaller_tlv_functor());
		tlv_intersection_2.resize( intersection_end2 - tlv_intersection_2.begin() );
		num_non_intersecting_tlvs -= tlv_intersection_2.size();

	}

	if (verbose) {
		std::cout << "tlvs:\n---------------\n";
		print_equations(tlvs_L, tlvs_R);
	}
	
		
	// -----------------------------------------------
	// partitioned communication step of intersections
	// communicate the items of intersections 1 and 2 pairwise
	// -----------------------------------------------
	if (tlv_intersection_1.size() > 0) {
		if (verbose) {
			std::cout << "tlv intersections (1): \n---------------------\n";
			print_equations(tlv_intersection_1);
			std::cout << " + (2) \n";
			print_equations(tlv_intersection_2);
		}

		// communication happens here!
		comm_functor cf;
		// transform allows in-place update
		thrust::transform(tlv_intersection_1.begin(), tlv_intersection_1.end(), tlv_intersection_2.begin(), tlv_intersection_1.begin(), cf);

		if (verbose) {
			std::cout << "partition communication result (intersection 1): \n---------------------\n";
			print_equations(tlv_intersection_1);
		}
	}


		

	// ----------------
	// partition update
	// ----------------

	//move non-intersecting tlvs to tlvs vector
	// ----------------------------------------
	//make a new vector for the non-intersecting tlvs
	//TODO this will break if non-intersecting tlvs contains two communication-eligible tlvs!
	if (num_non_intersecting_tlvs > 0) {
		thrust::device_vector<equation> non_intersecting_tlvs((aps_end - aps_partition) + tlv_intersection_2.size());
		//only do set difference if there are intersects
		if (tlv_intersection_2.size() > 0) {
			//TODO stuff crashes here
			// maybe size isnt right?
			thrust::sort(aps_partition, aps_end, smaller_tlv_functor());
			thrust::sort(tlv_intersection_2.begin(), tlv_intersection_2.end(), smaller_tlv_functor());
			if (verbose) {
				std::cout << "compute non-intersecting tlvs, loop "<< num_loops <<"\n";
				std::cout << "current aps: \n---------------------\n";
				print_equations(aps_L, aps_R);
				std::cout << " (intersection 2) \n---------------------\n";
				print_equations(tlv_intersection_2);
			}
			// set operation is fine, no in-place-update
			auto non_intersect_end = thrust::set_difference(aps_partition, aps_end, tlv_intersection_2.begin(), tlv_intersection_2.end(), non_intersecting_tlvs.begin(), smaller_tlv_functor());
			non_intersecting_tlvs.resize(non_intersect_end - non_intersecting_tlvs.begin());
			if (verbose) {
				std::cout << "non-intersecting tlvs: \n---------------------\n";
				print_equations(non_intersecting_tlvs);
			}
		}	
		//if there are no intersections, then it means that all tlvs from aps vector should go to tlvs vector straight
		else {
			if (verbose) {
				std::cout << "intersections empty, copying tlvs from aps straight\n";
			}
			thrust::copy(aps_partition, aps_end, non_intersecting_tlvs.begin());
		}
		//if (verbose) 
		//{
			//std::cout << "non-intersecting tlvs: \n---------------------\n";
			//print_equations(non_intersecting_tlvs);
			//std::cout << "tlvs before merge: \n---------------------\n";
			//print_equations(tlvs_L, tlvs_R);
		//}

		//merge into tlvs vector
		// TODO will copy keep tlv order? (assumption: only tlvs with new ids will be copied)
		int old_tlv_size = tlvs_L.size();
		tlvs_L.resize(old_tlv_size + non_intersecting_tlvs.size());
		tlvs_R.resize(old_tlv_size + non_intersecting_tlvs.size());
		tlvs_begin = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.begin(), tlvs_R.begin() ) );
		// replace merge by copy/sort
		// TODO triangle swap + merge might be faster
		tlvs_end = thrust::copy(non_intersecting_tlvs.begin(), non_intersecting_tlvs.end(), tlvs_begin + old_tlv_size);
		thrust::sort(tlvs_begin, tlvs_end, smaller_tlv_functor() );

		tlvs_end = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.end(), tlvs_R.end() ) );
		//if (verbose) {
			//std::cout << "merged tlvs: \n---------------------\n";
			//print_equations(tlvs_L, tlvs_R);
		//}

	}

	// move new APs to aps vector
	//partition comm result into aps and tlvs
	int num_new_actives = 0;
	if (tlv_intersection_1.size() > 0) {
		// copy is not stable, but APs dont need to be sorted
		thrust::copy(tlv_intersection_1.begin(), tlv_intersection_1.end(), aps_partition);
		num_new_actives = tlv_intersection_1.size();
		if (verbose) {
			std::cout << "partition: number of new aps:" << num_new_actives << "\n";
		}
	}
	int num_non_aps = aps_end - aps_partition;
	aps_L.resize(aps_L.size() - (num_non_aps - num_new_actives));
	aps_R.resize(aps_R.size() - (num_non_aps - num_new_actives));
	ap_size = aps_L.size();

	if (verbose) {
		std::cout << "new ap partition: \n---------------------\n";
		print_equations(aps_L, aps_R);
	}

	if (verbose) {
		std::cout << "old tlv partition: \n---------------------\n";
		print_equations(tlvs_L, tlvs_R);
		//std::cout << "second tlv intersection: \n---------------------\n";
		//print_equations(tlv_intersection_2);
	}

	// move new TLVs to tlvs vector
	// only if there are intersections
	if (tlv_intersection_2.size() > 0) {
		//resize tlvs to maximum possible difference
		int tlvs_old_size = tlvs_L.size();
		tlvs_L.resize( tlvs_old_size + tlv_intersection_2.size() );
		tlvs_R.resize( tlvs_R.size() + tlv_intersection_2.size() );
		//update iterators
		tlvs_begin = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.begin(), tlvs_R.begin() ) );
		tlvs_end = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.end(), tlvs_R.end() ) );
		if (verbose) {
				std::cout << " (intersection 2), loop "<< num_loops <<"  \n---------------------\n";
				print_equations(tlv_intersection_2);
		}
		//assert( thrust::is_sorted(tlvs_begin, tlvs_end, smaller_tlv_functor() ) );
		//assert(thrust::is_sorted(tlv_intersection_2.begin(), tlv_intersection_2.end(), smaller_tlv_functor() ) );
		//if (verbose) {
		//	thrust::device_vector<equation> test(tlvs_R.size());
		//	std::cout << "doing the test\n";
		//	thrust::copy(tlvs_begin, tlvs_begin + tlvs_old_size, test.begin());
		//	std::cout << "copied tlvs\n";
		//	thrust::copy(tlv_intersection_2.begin(), tlv_intersection_2.end(), test.begin());
		//	std::cout << "copied intersections\n";
		//	std::cout << "sorted stuff\n";
		//	auto test_end = thrust::set_union(tlvs_begin, tlvs_begin + tlvs_old_size, tlv_intersection_2.begin(), tlv_intersection_2.end(), test.begin(), smaller_tlv_functor());
		//	std::cout << " (symmetric difference test 2) \n---------------------\n";
		//	print_equations(test);
		//}
		thrust::sort(tlvs_begin, tlvs_begin + tlvs_old_size, smaller_tlv_functor());
		//TODO why is intersection_2 not sorted here?
		thrust::sort(tlv_intersection_2.begin(), tlv_intersection_2.end(), smaller_tlv_functor());
		//TODO remove in-place update
		//auto tlvs_new_end = thrust::set_symmetric_difference(tlvs_begin, tlvs_begin + tlvs_old_size, tlv_intersection_2.begin(), tlv_intersection_2.end(), tlvs_begin, smaller_tlv_functor());
		// get an auxiliary vector as large as tlvs and intersection 2 together
		thrust::device_vector<equation> tlvs_symmetric_diff(tlvs_L.size());
		auto tlvs_new_end = thrust::set_difference(tlvs_begin, tlvs_begin + tlvs_old_size, tlv_intersection_2.begin(), tlv_intersection_2.end(), tlvs_symmetric_diff.begin(), smaller_tlv_functor());
		int tlvs_new_size = tlvs_new_end - tlvs_symmetric_diff.begin();
		//swap tlvs vector and auxiliary vector
		//TODO could also use intersection 1 here to save memory?
		StopWatch::get(communication_watch).start();
		thrust::swap_ranges(tlvs_symmetric_diff.begin(), tlvs_new_end, tlvs_begin);
		StopWatch::get(communication_watch).stop();
		// the symmetric difference should now be in tlvs_L/tlvs_R

		//resize tlvs
		tlvs_L.resize(tlvs_new_size);
		tlvs_R.resize(tlvs_new_size);
		if (verbose) {
			std::cout << "new tlv partition: \n---------------------\n";
			print_equations(tlvs_L, tlvs_R);
		}
	}

	// iterator update
	aps_begin = thrust::make_zip_iterator( thrust::make_tuple(aps_L.begin(), aps_R.begin() ) );
	aps_end = thrust::make_zip_iterator( thrust::make_tuple(aps_L.end(), aps_R.end() ) );
	tlvs_begin = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.begin(), tlvs_R.begin() ) );
	tlvs_end = thrust::make_zip_iterator( thrust::make_tuple(tlvs_L.end(), tlvs_R.end() ) );

	//StopWatch::get(interaction_watch).stop();

	//update number of loops
	num_loops++;

	//maximum index update
	int new_maximum_index = dev_ptr[0];
	if (new_maximum_index < maximum_index) {
		std::cout << "error: index overflow: " << new_maximum_index << " " << maximum_index << std::endl;
		break;
	}
	maximum_index = new_maximum_index;
	
	//StopWatch::get(communication_watch).stop();
	//partition termination condition
	if (ap_size < 1) {
		if (verbose) {
			
			std::cout << "------------------------\nap partition empty ("<< ap_size <<") , were done!\n------------------------\n";
		}
		break;
	}
	else {
		//resize additional result vectors
		a_L.clear(); a_R.clear(); 
		a_L.resize(ap_size); a_R.resize(ap_size);
		b_L.clear(); b_R.clear();
		b_L.resize(ap_size); b_R.resize(ap_size);
		c_L.clear(); c_R.clear();
		c_L.resize(ap_size); c_R.resize(ap_size);
	}

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
	// -------------------------
	// END OF LOOP
	// -------------------------

	//if (!verbose) {
	if (print_final_result) {
		std::cout << "aps:\n";
		print_equations(aps_L, aps_R);
		std::cout << "tlvs:\n";
		print_equations(tlvs_L, tlvs_R);
	}
	////compute result for addition tests
	//int s = thrust::count_if(result_rhs.begin(), result_rhs.end(), has_name('S'));
	int s_part = thrust::count_if(tlvs_R.begin(), tlvs_R.end(), has_name('S'));
	
	//highest index:
	std::cout << "maximum index used: " << maximum_index << std::endl;
	// stop and destroy timer
	float int_time = StopWatch::get(interaction_watch).getTime() / 1000.0;
	float com_time = StopWatch::get(communication_watch).getTime() / 1000.0;
	printf ("total interaction time %.5f seconds.\n", int_time );
	printf ("total communication time %.5f seconds.\n", com_time );
	StopWatch::destroy(communication_watch);
	StopWatch::destroy(interaction_watch);

	//free index memory
	thrust::device_free(dev_ptr);

	return result_info(s_part, num_loops, total_interactions_partition, com_time, int_time);
	
}


//testing function
result_info test_interaction_step(const int ruleset, 
	thrust::host_vector<Agent>& in_lhs_host, thrust::host_vector<Agent>& in_rhs_host, const bool verbose, const bool print_final_result, const bool pause)
{
	// ---------
	// constants
	// ---------
	//input size
	int input_size = in_lhs_host.size();

	//make a device pointer for tracking agent indices
	//------------------------------------------
	const size_t ptr_size = 1;
	//create a dev ptr
	thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(ptr_size);
	//TODO: ugly hardcoded intial index again!
	dev_ptr[0] = 100;
	int maximum_index = 100;

	std::cout << "ruleset: " << ruleset << std::endl;
	std::cout << "done allocating\n";
	//extract the raw pointer
	int * index_ptr = thrust::raw_pointer_cast(dev_ptr);


	// transfer the input to the device
	//---------------------------------
	thrust::device_vector<Agent> in_lhs(in_lhs_host);
	thrust::device_vector<Agent> in_rhs(in_rhs_host);
	//aux vectors for communication
	thrust::device_vector<Agent> aux_lhs(input_size);
	thrust::device_vector<Agent> aux_rhs(input_size);


	//track whether an interaction step was made
	thrust::device_vector<bool> normal(input_size);
	
	//result vectors
	thrust::host_vector<Agent> result_lhs(in_lhs.size());
	thrust::host_vector<Agent> result_rhs(in_rhs.size());

	// build the additional results equation array
	// --------------------------------------
	if (verbose) {
		std::cout << "input (size " << input_size << " ) \n--------\n";
	}
	thrust::device_vector<Agent> a_lhs(input_size);
	thrust::device_vector<Agent> a_rhs(input_size);
	thrust::device_vector<Agent> b_lhs(input_size);
	thrust::device_vector<Agent> b_rhs(input_size);
	thrust::device_vector<Agent> c_lhs(input_size);
	thrust::device_vector<Agent> c_rhs(input_size);

	if (verbose) {
		print_equations(in_lhs, in_rhs);
	}

	//loop variable
	int num_interactions = 0;
	int num_loops = 0;
	int total_interactions = 0;
	int total_interactions_partition = 0;

	//functor
	ackermann_functor ack_functor(index_ptr);
	//add_functor functor;
	//ic_functor functor2;
	fibonacci_functor fib_functor(index_ptr);
	algae_functor alg_functor(index_ptr);
	
	//dummy test
	valid_tuple_functor dummy_test;
	//sort by key functor
	get_tlv_single_functor key_functor;
	//clearing auxiliary vectors
	clear_functor clear;

	//stopwatches
	const unsigned int interaction_watch = StopWatch::create();
	const unsigned int communication_watch = StopWatch::create();

	//result info
	result_info info(0,0,0);

	// ---------------
	// start of loop
	// ---------------
	do {
	if (verbose) {
		std::cout << "regular input " << input_size << ":\n";
		print_equations(in_lhs, in_rhs);
	}

	//original int start position
	//StopWatch::get(interaction_watch).start();
	// -------------------------------------------
	// apply interaction step to list of equations
	// -------------------------------------------

	//if (verbose) {
	//	print_range("normal_before", normal.begin(), normal.end());
	//}
	switch (ruleset) {
	case 1:
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(in_lhs.begin(), in_rhs.begin(), a_lhs.begin(), a_rhs.begin(), b_lhs.begin(), b_rhs.begin(), c_lhs.begin(), c_rhs.begin(), normal.begin() ) ),
			thrust::make_zip_iterator(thrust::make_tuple(in_lhs.end(),   in_rhs.end(),   a_lhs.end(),   a_rhs.end(),   b_lhs.end(),   b_rhs.end(),   c_lhs.end(),   c_rhs.end(),   normal.end()  ) ),
			ack_functor
		);
		break;
	case 2:
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(in_lhs.begin(), in_rhs.begin(), a_lhs.begin(), a_rhs.begin(), b_lhs.begin(), b_rhs.begin(), c_lhs.begin(), c_rhs.begin(), normal.begin() ) ),
			thrust::make_zip_iterator(thrust::make_tuple(in_lhs.end(),   in_rhs.end(),   a_lhs.end(),   a_rhs.end(),   b_lhs.end(),   b_rhs.end(),   c_lhs.end(),   c_rhs.end(),   normal.end()  ) ),
			alg_functor
		);
		break;
	case 3:
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(in_lhs.begin(), in_rhs.begin(), a_lhs.begin(), a_rhs.begin(), b_lhs.begin(), b_rhs.begin(), c_lhs.begin(), c_rhs.begin(), normal.begin() ) ),
			thrust::make_zip_iterator(thrust::make_tuple(in_lhs.end(),   in_rhs.end(),   a_lhs.end(),   a_rhs.end(),   b_lhs.end(),   b_rhs.end(),   c_lhs.end(),   c_rhs.end(),   normal.end()  ) ),
			fib_functor
		);
		break;
	}

	//------------------------------------------------
	// end of interaction step
	//------------------------------------------------
	//StopWatch::get(interaction_watch).start();
	//StopWatch::get(interaction_watch).stop();
	//check if we performed one or more interactions
	//int_normal_form = thrust::any_of(normal.begin(), normal.end(), thrust::identity<bool>());
	//StopWatch::get(interaction_watch).start();
	num_interactions = thrust::count(normal.begin(), normal.end(), true);
	if (num_interactions > 0) { 
		num_loops++;
		if (verbose) {
		//print_range("normal_after", normal.begin(), normal.end());
		std::cout << "interactions: " << num_interactions << "\n"; 
		}
	}
	//else {
	//	if (verbose) {
	//		std::cout << "equations are in int normal form\n\n";
	//	}
	//	break;
	//}
	
	// merge equation arrays
	// TODO less copy operations
	// ---------------------

	in_lhs.resize(input_size * 4);
	in_rhs.resize(input_size * 4);

	equation_iter orient_begin = thrust::make_zip_iterator( thrust::make_tuple(in_lhs.begin(), in_rhs.begin() ) );
	equation_iter orient_end = thrust::make_zip_iterator( thrust::make_tuple(in_lhs.end(), in_rhs.end() ) );

	// append A to IN
	orient_end = thrust::copy_if(thrust::make_zip_iterator( thrust::make_tuple(a_lhs.begin(), a_rhs.begin() ) ),
								thrust::make_zip_iterator( thrust::make_tuple(a_lhs.end(), a_rhs.end() ) ),
								orient_begin + input_size, dummy_test);
	//thrust::copy(a_lhs.begin(), a_lhs.end(), in_lhs.begin() + input_size);
	//thrust::copy(a_rhs.begin(), a_rhs.end(), in_rhs.begin() + input_size);

	// append B to IN
	orient_end = thrust::copy_if(thrust::make_zip_iterator( thrust::make_tuple(b_lhs.begin(), b_rhs.begin() ) ),
								thrust::make_zip_iterator( thrust::make_tuple(b_lhs.end(), b_rhs.end() ) ),
								orient_end, dummy_test);
	//thrust::copy(b_lhs.begin(), b_lhs.end(), in_lhs.begin() + input_size*2);
	//thrust::copy(b_rhs.begin(), b_rhs.end(), in_rhs.begin() + input_size*2);
	// append C to IN
	orient_end = thrust::copy_if(thrust::make_zip_iterator( thrust::make_tuple(c_lhs.begin(), c_rhs.begin() ) ),
								thrust::make_zip_iterator( thrust::make_tuple(c_lhs.end(), c_rhs.end() ) ),
								orient_end, dummy_test);
	//thrust::copy(c_lhs.begin(), c_lhs.end(), in_lhs.begin() + input_size*3);
	//thrust::copy(c_rhs.begin(), c_rhs.end(), in_rhs.begin() + input_size*3);
	
	//std::cout << "\n oriented result\n-----------\n";
	
	//remove dummy equations
	//equation_iter new_end = thrust::remove_if(
	//	orient_begin,
	//	orient_end,
	//	dummy_test_functor_tuple()
	//	);

	input_size = orient_end - orient_begin;
	in_lhs.resize(input_size);
	in_rhs.resize(input_size);

	//if (new_end != orient_end) {
	//	input_size = new_end - orient_begin;
	//	orient_end = new_end;
	//	in_lhs.resize(input_size);
	//	in_rhs.resize(input_size);
	//}

	
	if (verbose) {
		std::cout << "after dummy removal:\n---------------\n";
		print_equations(in_lhs, in_rhs);
	}
	
	//StopWatch::get(interaction_watch).stop();
	
	StopWatch::get(interaction_watch).start();
	// only perform communication on tlv equations
	//auto tlv_iter = thrust::partition(orient_begin, orient_end, is_active_functor());
	equation_iter tlv_iter = orient_begin;
	//aps generated by the int step
	int num_aps = tlv_iter - orient_begin;
	//number of tlvs
	int num_tlvs = orient_end - tlv_iter;


	thrust::transform_iterator<get_tlv_single_functor, agent_iter> keys_begin(in_lhs.begin() + num_aps, key_functor);
	thrust::transform_iterator<get_tlv_single_functor, agent_iter> keys_end(in_lhs.end(), key_functor);

	//get the permutation
	thrust::device_vector<int> permutation(num_tlvs);
	thrust::sequence(permutation.begin(), permutation.end());

	thrust::sort_by_key(keys_begin, keys_end, permutation.begin());

	aux_lhs.resize(num_tlvs);
	aux_rhs.resize(num_tlvs);
	equation_iter aux_begin = thrust::make_zip_iterator( thrust::make_tuple(aux_lhs.begin(), aux_rhs.begin() ) );
	equation_iter aux_end = thrust::make_zip_iterator( thrust::make_tuple(aux_lhs.end(), aux_rhs.end() ) );

	thrust::gather(permutation.begin(), permutation.end(), orient_begin + num_aps, aux_begin);

	//thrust::stable_sort_by_key(keys_begin, keys_end, orient_begin)

	if (verbose) {
		result_lhs = in_lhs;
		result_rhs = in_rhs;
		std::cout << "\n sorted and oriented result\n-----------\n";
		print_equations(result_lhs, result_rhs);
		print_equations(aux_begin, aux_end);
	}

	//original int watch position
	StopWatch::get(interaction_watch).stop();

	/*-------------------------
	/ communication starts here
	---------------------------*/

	// parallel communication
	
	StopWatch::get(communication_watch).start();
	//aux vectors to remove in-place udpate

	//one iteration of parallel communication
	int steps = comm_step_parallel( aux_begin, aux_end, tlv_iter, orient_end, num_tlvs, num_loops, communication_watch);
	// normal form condition
	if ((steps == 0) && (num_interactions == 0)) {
		if (verbose) {
			std::cout << "no more communications, normal form reached\n";
		}
		break;
	}
	int comm_size = orient_end - orient_begin;
	if (comm_size < input_size) {
		in_lhs.resize(comm_size);
		in_rhs.resize(comm_size);
		input_size = comm_size;
	}
	if (verbose) {
		result_lhs = in_lhs;
		result_rhs = in_rhs;
		std::cout << "\n parallel communication result ( "<< steps << " steps) \n-----------\n";
		print_equations(result_lhs, result_rhs);

		print_equations(orient_begin, orient_end);
	}
		
	StopWatch::get(communication_watch).stop();
	/*-------------------------
	/ communication ends here
	---------------------------*/
	
	//maximum index update
	int new_maximum_index = dev_ptr[0];
	if (new_maximum_index < maximum_index) {
		std::cout << "error: index overflow: " << new_maximum_index << " " << maximum_index << std::endl;
		break;
	}
	//maximum_index = new_maximum_index;

	//clear result vectors
	//TODO this is probably slow, can it be done in one for_each transform?
	//a_lhs.clear(); a_rhs.clear(); 
	a_lhs.resize(input_size); a_rhs.resize(input_size);
	//b_lhs.clear(); b_rhs.clear();
	b_lhs.resize(input_size); b_rhs.resize(input_size);
	//c_lhs.clear(); c_rhs.clear();
	c_lhs.resize(input_size); c_rhs.resize(input_size);
	//normal.clear();
	normal.resize(input_size);
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(a_lhs.begin(), a_rhs.begin(), b_lhs.begin(), b_rhs.begin(), c_lhs.begin(), c_rhs.begin(), normal.begin() ) ),
			thrust::make_zip_iterator(thrust::make_tuple(a_lhs.end(),   a_rhs.end(),   b_lhs.end(),   b_rhs.end(),   c_lhs.end(),   c_rhs.end(),   normal.end()  ) ),
			clear_functor()
		);

	//interaction statisitics
	total_interactions += num_interactions;
	//store interactions per loop
	//info.loops_per_step.push_back(num_interactions);

	if (verbose) {
		std::cout << "\n-------------------------------------\n";
		std::cout << "end of interaction loop " << num_loops <<"\n";
		std::cout << "-------------------------------------\n";
		if (pause) {
			std::cout << "press a key to continue\n";
			getch();
		}
	}
	} while (true);

	//if (!verbose) {
	if (print_final_result) {
		result_lhs = in_lhs;
		result_rhs = in_rhs;
		std::cout << "\n result\n-----------\n";
		print_equations(result_lhs, result_rhs);
	}
	//}
	//std::cout << "total loops: " << num_loops << std::endl;
	//std::cout << "total interactions: " << total_interactions << std::endl;
	//const float avg_interactions = static_cast<float>(total_interactions) / static_cast<float>(num_loops);
	//std::cout << "average interactions per loop: " << avg_interactions << std::endl;
	//
	////compute result for addition tests
	int s = 0;
	s = thrust::count_if(in_rhs.begin(), in_rhs.end(), has_name('S'));
	std::cout << "total interactions: " << total_interactions_partition << std::endl;
	
	//highest index:
	std::cout << "maximum index used: " << dev_ptr[0] << std::endl;
	// stop and destroy timer
	float int_time = StopWatch::get(interaction_watch).getTime() / 1000.0;
	float com_time = StopWatch::get(communication_watch).getTime() / 1000.0;
	printf ("total interaction time %.5f seconds.\n", int_time );
	printf ("total communication time %.5f seconds.\n", com_time );
	StopWatch::destroy(communication_watch);
	StopWatch::destroy(interaction_watch);

	//free index memory
	thrust::device_free(dev_ptr);

	info.result = s;
	info.num_loops = num_loops;
	info.total_interactions = total_interactions;
	info.com_time = com_time;
	info.int_time = int_time;

	return info;
	
}